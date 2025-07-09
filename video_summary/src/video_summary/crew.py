from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import FileWriterTool
from typing import List
import yt_dlp
import urllib.parse
import json
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled, VideoUnavailable
import os
from crewai_tools import SerperDevTool
from crewai.tools import tool
from dotenv import load_dotenv
import torch
from transformers import pipeline

STT_MODEL_ID = "ian-r3/stt_model"
transcription_pipeline = None
def get_transcription_pipeline():
    global transcription_pipeline
    if transcription_pipeline is None:
        device = 0 if torch.cuda.is_available() else -1
        transcription_pipeline = pipeline(
            "automatic-speech-recognition",
            model=STT_MODEL_ID,
            device=device,
            chunk_length_s=30,
        )
    return transcription_pipeline

# Dynamic FFmpeg path detection
def setup_ffmpeg_path():
    """Setup FFmpeg path dynamically for different environments"""
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Look for ffmpeg in various possible locations
    possible_paths = [
        # Relative to current script (for development)
        os.path.join(current_dir, "..", "..", "ffmpeg", "bin"),
        # Relative to project root
        os.path.join(current_dir, "..", "..", "..", "ffmpeg", "bin"),
        # Environment variable
        os.environ.get("FFMPEG_PATH"),
        # System PATH (will be checked automatically)
        None
    ]
    
    # Try to find ffmpeg
    for path in possible_paths:
        if path and os.path.exists(path):
            # print(f"Found FFmpeg at: {path}")
            os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
            return path
    
    # If no local ffmpeg found, check if it's in system PATH
    try:
        import subprocess
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            # print("FFmpeg found in system PATH")
            return "system"
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    
    # print("Warning: FFmpeg not found. Please ensure FFmpeg is installed and in PATH")
    return None

# Setup FFmpeg path
ffmpeg_path = setup_ffmpeg_path()

# Use default device detection (Whisper will choose the best available device)
DEVICE = None
# print("Using default device detection for transcription")

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
        # print(f"Testing Whisper transcription with file: {audio_file_path}")
        # print(f"File exists: {os.path.exists(audio_file_path)}")

        pipe = get_transcription_pipeline()
        result = pipe(audio_file_path)
        
        return result["text"]
    except Exception as e:
        return f"Error during transcription: {str(e)}"

@tool("Audio Transcribe Tool")
def audio_transcriber_tool(input_str: str) -> str:
    """
    Extracts transcript from a YouTube video given its URL or transcribes an audio file.
    Uses YouTube's transcript API for YouTube videos or Whisper for audio files.

    Parameters:
    - input_str (str): A JSON string containing either a YouTube URL or audio file path.

    Returns:
    str: The transcribed text from the YouTube video or audio file.
    """
    # print(f"Received input: {input_str}")
    
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
            # print(f"Received transcript: {transcript_data}")
            return ' '.join([snippet.text for snippet in transcript_data.snippets])
        except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable):
            return None

    def is_youtube_url(text: str) -> bool:
        """Check if the input is a YouTube URL"""
        return 'youtube.com' in text or 'youtu.be' in text

    try:
        if input_str.strip().startswith('{'):
            inputs = json.loads(input_str)
            content = inputs.get('content') or inputs.get('url') or inputs.get('input_str') or inputs.get('youtube_url') or inputs.get('audio_file_path')
            if content is None:
                raise ValueError("Content is required in the input JSON.")
        else:
            content = input_str.strip()
            if not content:
                raise ValueError("Input content is empty.")

        # Check if it's a YouTube URL or file path
        if is_youtube_url(content):
            # print(f"Processing YouTube URL: {content}")
            # Get transcript from YouTube
            youtube_transcription = get_youtube_transcription(content)
            if youtube_transcription:
                return youtube_transcription

            # If no subtitles, proceed with Whisper transcription
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
                ydl.download([content])

            audio_file = "audio_file.mp3"
            pipe = get_transcription_pipeline()
            result = pipe(audio_file)
            # print(f"Transcription completed")

            os.remove(audio_file)
            return result["text"]
        else:
            # Treat as audio file path
            # print(f"Processing audio file: {content}")
            if not os.path.exists(content):
                return f"Error: File not found at {content}"

            pipe = get_transcription_pipeline()
            result = pipe(content)
            
            return result["text"]
            
    except Exception as e:
        return f"Error downloading or transcribing audio: {e}"

@tool("Audio File Transcribe Tool")
def audio_file_transcriber_tool(file_path: str) -> str:
    """
    Transcribes an audio file to text using Whisper.

    Parameters:
    - file_path (str): The path to the audio file to be transcribed.

    Returns:
    str: The transcribed text of the audio file.
    """
    try:
        # print(f"Transcribing audio file: {file_path}")
        if not os.path.exists(file_path):
            return f"Error: File not found at {file_path}"

        pipe = get_transcription_pipeline()
        result = pipe(file_path)
        
        return result["text"]
    except Exception as e:
        return f"Error during file transcription: {str(e)}"

@CrewBase
class VideoSummary():
    """VideoSummary crew"""
    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self):
        self.audio_tool = [audio_transcriber_tool, audio_file_transcriber_tool]
        self.summaryReport = ""

    @agent
    def transcriber(self) -> Agent:
        return Agent(
            config=self.agents_config['transcriber'], 
            tools=self.audio_tool,
            verbose=True, 
            allow_delegation=False)

    @agent
    def summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config['summarizer'], 
            tools=[], 
            verbose=True, 
            allow_delegation=False)

    @agent
    def router_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['router_agent'], 
            tools=[], 
            verbose=True, 
            allow_delegation=True)

    @agent
    def responder_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['responder_agent'], 
            tools=[], 
            verbose=True, 
            allow_delegation=False)


    @agent
    def chat_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['chat_agent'], 
            tools=[], 
            verbose=True, 
            allow_delegation=True)
    @agent
    def info_finder(self) -> Agent:
        return Agent(
            config=self.agents_config['info_finder'],
            tools=[SerperDevTool()],
            verbose=True
        )


    @agent
    def filewriter(self) -> Agent:
        return Agent(
            config=self.agents_config['filewriter'], 
            tools=[FileWriterTool(file_path='Video_Summary.txt')], 
            verbose=True)

    @task
    def transcription_task(self) -> Task:
        return Task(
            config=self.tasks_config['transcription_task'], 
            tools=self.audio_tool)

    @task
    def summary_task(self) -> Task:
        return Task(
            config=self.tasks_config['summary_task'], 
            tools=[])

    @task
    def file_write_task(self) -> Task:
        return Task(
            config=self.tasks_config['file_write_task'])

    @task
    def chat_task(self) -> Task:
        return Task(
            config=self.tasks_config['chat_task'])
    @task
    def info_task(self) -> Task:
        return Task(
            config=self.tasks_config['info_task']
        )

    def create_summarization_crew(self) -> Crew:
        """
        Creates the crew responsible for transcription, summarization, and file writing.
        """
        return Crew(
            agents=[self.transcriber(), self.summarizer(), self.filewriter()],
            tasks=[self.transcription_task(), self.summary_task(), self.file_write_task()],
            process=Process.sequential,
            verbose=True,
        )

    def create_chat_crew(self) -> Crew:
        """
        Creates the crew responsible for handling chat interactions.
        This crew is hierarchical, with a chat agent managing an info finder.
        """
        # Instantiate the agents that will be part of the crew
        chat_agent_manager = self.chat_agent()
        info_finder_agent = self.info_finder()

        return Crew(
            # The 'agents' list should ONLY contain the worker agents.
            # The manager is defined separately and should not be in this list.
            agents=[info_finder_agent],
            tasks=[self.chat_task(), self.info_task()],
            process=Process.hierarchical,
            manager_agent=chat_agent_manager,
            verbose=False # Keep UI clean. Set to 2 to see delegation steps in your terminal.
        )