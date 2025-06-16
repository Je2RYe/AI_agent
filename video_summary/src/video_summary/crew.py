from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import FileWriterTool
from typing import List
import yt_dlp
import urllib.parse
import json
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled, VideoUnavailable
import whisper
import os
from crewai.tools import tool
from dotenv import load_dotenv

ffmpeg_bin_path = r"D:\Work\AI_agent\video_summary\ffmpeg\bin"
os.environ["PATH"] += os.pathsep + ffmpeg_bin_path

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators
load_dotenv()

# Define FFmpeg path relative to the current script
def test_whisper_transcription(audio_file_path: str) -> str:
    """
    Test function to verify if Whisper can transcribe audio.
    
    Args:
        audio_file_path (str): Path to the audio file to transcribe
        
    Returns:
        str: Transcribed text or error message
    """
    try:
        print(f"Testing Whisper transcription with file: {audio_file_path}")
        print(f"File exists: {os.path.exists(audio_file_path)}")
        
        whisper_model = whisper.load_model("small")
        print("Whisper model loaded successfully")
        
        print("Starting transcription...")
        result = whisper_model.transcribe(audio_file_path)
        print("Transcription completed successfully")
        
        return result["text"]
    except Exception as e:
        return f"Error during transcription: {str(e)}"

@tool("Audio Transcribe Tool")
def audio_transcriber_tool(input_str: str) -> str:
    """
    Extracts transcript from a YouTube video given its URL.
    Uses YouTube's transcript API to get the video's transcript.

    Parameters:
    - input_str (str): A JSON string containing the URL of the YouTube video.

    Returns:
    str: The transcribed text from the YouTube video.
    """
    print(f"Received input: {input_str}")
    def extract_video_id(url):
        parsed_url = urllib.parse.urlparse(url)
        hostname = parsed_url.hostname.lower() if parsed_url.hostname else ''
        if 'youtu.be' in hostname:
            return parsed_url.path[1:]
        elif 'youtube.com' in hostname:
            if parsed_url.path == '/watch':
                query = urllib.parse.parse_qs(parsed_url.query)
                return query.get('v', [None])[0]
            elif parsed_url.path.startswith(('/embed/', '/v/')):
                return parsed_url.path.split('/')[2]
        return None

    def get_youtube_transcription(url: str) -> str:
        video_id = extract_video_id(url)
        if not video_id:
            return None

        try:
            transcript_data = YouTubeTranscriptApi().fetch(video_id, languages=['fr', 'en'])
            print(f"Received transcript: {transcript_data}")
            return ' '.join([snippet.text for snippet in transcript_data.snippets])
        except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable):
            return None

    try:
        if input_str.strip().startswith('{'):
            inputs = json.loads(input_str)
            url = inputs.get('url') or inputs.get('input_str') or inputs.get('youtube_url')
            if url is None:
                raise ValueError("URL is required in the input JSON.")
        else:
            url = input_str.strip()
            if not url:
                raise ValueError("Input URL is empty.")

        # # Get transcript from YouTube
        # youtube_transcription = get_youtube_transcription(url)
        # if youtube_transcription:
        #     return youtube_transcription

        # Step 2: If no subtitles, proceed with Whisper transcription
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': 'audio_file.%(ext)s',
            'quiet': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        audio_file = "audio_file.mp3"
        whisper_model = whisper.load_model("small")
        result = whisper_model.transcribe(audio_file)

        os.remove(audio_file)
        return result["text"]
    except Exception as e:
        return f"Error downloading or transcribing audio: {e}"

@CrewBase
class VideoSummary():
    """VideoSummary crew"""

    agents: List[BaseAgent]
    tasks: List[Task]
    def __init__(self):
        # Load audio transcriber tool
        self.audio_tool = [audio_transcriber_tool]
        #summary generated
        self.summaryReport=""
    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def transcriber(self) -> Agent:
        return Agent(
            config=self.agents_config['transcriber'],
            tools=self.audio_tool,  # Uses audio transcriber tool for converting podcast audio to text.
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config['summarizer'],
            tools=[],  # No specific tools required, relies on LLM for summarization.
            verbose=True,
            allow_delegation=False,
        )
    @agent
    def filewriter(self) -> Agent:
        return Agent(
            config=self.agents_config['filewriter'],
            tools=[FileWriterTool()],
            verbose=True
        )


    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def transcription_task(self) -> Task:
        return Task(
            config=self.tasks_config['transcription_task'],
            tools=self.audio_tool,  # Uses audio transcriber tool for converting audio to text.
        )

    @task
    def summary_task(self) -> Task:
        return Task(
            config=self.tasks_config['summary_task'],
            tools=[],  # No specific tools needed.
        )
    @task
    def file_write_task(self) -> Task:
        return Task(
            config=self.tasks_config['file_write_task'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the VideoSummary crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )