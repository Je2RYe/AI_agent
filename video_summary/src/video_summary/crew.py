from crewai import Agent, Crew, Process, Task,LLM
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
from crewai_tools import SerperDevTool
from crewai.tools import tool
from dotenv import load_dotenv

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
        
        whisper_model = whisper.load_model("tiny", device=DEVICE)
        # print(f"Whisper model loaded successfully")
        
        # print("Starting transcription...")
        result = whisper_model.transcribe(audio_file_path)
        # print(f"Transcription completed successfully")
        
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
            transcript_data = YouTubeTranscriptApi().fetch(video_id, languages=['en','fr'])
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
            whisper_model = whisper.load_model("small", device=DEVICE)
            result = whisper_model.transcribe(audio_file)
            # print(f"Transcription completed")

            os.remove(audio_file)
            return result["text"]
        else:
            # Treat as audio file path
            # print(f"Processing audio file: {content}")
            if not os.path.exists(content):
                return f"Error: File not found at {content}"
            
            whisper_model = whisper.load_model("small", device=DEVICE)
            # print(f"Whisper model loaded successfully for file transcription.")
            
            # print("Starting transcription of file...")
            result = whisper_model.transcribe(content)
            # print(f"File transcription completed successfully.")
            
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
        
        whisper_model = whisper.load_model("small", device=DEVICE)
        # print(f"Whisper model loaded successfully for file transcription.")
        
        # print("Starting transcription of file...")
        result = whisper_model.transcribe(file_path)
        # print(f"File transcription completed successfully.")
        
        return result["text"]
    except Exception as e:
        return f"Error during file transcription: {str(e)}"

@CrewBase
class VideoSummary():
    """VideoSummary crew"""
    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self, model_name="gemini-2.0-flash-lite-001"):
        # 1. 初始化工具
        self.audio_tool = [audio_transcriber_tool, audio_file_transcriber_tool]
        self.summaryReport = ""
        
        self.model_name = model_name
        
        # 2. 获取 API Key
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
            
        # 3. 核心修改：处理模型名称前缀
        # 如果传入的名称没有 gemini/ 前缀，手动添加
        # 这是 LiteLLM 识别 Google Gemini 的标准格式
        if not self.model_name.startswith("gemini/"):
            self.model_name = f"gemini/{self.model_name}"

        # 4. 使用 CrewAI 原生 LLM 类替代 LangChain
        # 4. 初始化核心自适应 LLM (用于 Summarizer, Chat)
        self.llm = LLM(
            model=self.model_name,
            api_key=api_key,
            temperature=0.7
        )
        
        # [新增 1] 专门用于低开销任务的 LLM 实例 (固定 Lite 模型)
        self.low_cost_llm = LLM(
            model="gemini/gemini-2.0-flash-lite-001",
            api_key=api_key,
            temperature=0.1 # 低温，更稳定
        )
        
        # [新增 2] 专门用于审计的高性能 LLM (固定 Pro 模型)
        self.audit_llm = LLM(
             model="gemini/gemini-2.5-flash", # 强制使用 Pro
             api_key=api_key,
             temperature=0.0 # 零度，确保客观性
        )
        
        super().__init__()

    # 修改 Agent 定义，显式传入 llm=self.llm
    @agent
    def transcriber(self) -> Agent:
        return Agent(
            config=self.agents_config['transcriber'], 
            tools=self.audio_tool,
            verbose=True, 
            allow_delegation=False,
            llm=self.low_cost_llm  
        )

    @agent #自适应
    def summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config['summarizer'], 
            tools=[], 
            verbose=True, 
            allow_delegation=False,
            llm=self.llm 
        )

    @agent #自适应
    def router_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['router_agent'], 
            tools=[], 
            verbose=True, 
            allow_delegation=True,
            llm=self.llm 
        )

    @agent #自适应
    def responder_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['responder_agent'], 
            tools=[], 
            verbose=True, 
            allow_delegation=False,
            llm=self.llm 
        )

    @agent #自适应
    def chat_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['chat_agent'], 
            tools=[], 
            verbose=True, 
            allow_delegation=True,
            llm=self.llm 
        )
    
    @agent 
    def info_finder(self) -> Agent:
        return Agent(
            config=self.agents_config['info_finder'],
            tools=[SerperDevTool()],
            verbose=True,
            llm=self.low_cost_llm
        )
    @agent
    def filewriter(self) -> Agent:
        return Agent(
            config=self.agents_config['filewriter'], 
            tools=[FileWriterTool(file_path='Video_Summary.txt')], 
            verbose=True,
            llm=self.low_cost_llm
        )
    @agent
    def evaluator(self) -> Agent:
        return Agent(
            role="Senior Quality Assurance Specialist",
            goal="Objectively evaluate the quality of the generated summary against strict standards.",
            backstory="""You are a strict, objective AI auditor. Your job is to grade summaries 
            based on a fixed 10-point framework. You are not afraid to give low scores 
            if the work is poor. You ignore cost and speed; you only care about the text quality.""",
            verbose=True,
            allow_delegation=False,
            # 强制使用高性能模型进行审查，确保评分质量
            llm=self.audit_llm
        )
    @task
    def evaluation_task(self) -> Task:
        return Task(
            description="""
            You are a RUTHLESS, objective "AI Auditor". Your job is to actively find flaws in the provided 'Generated Summary'. 
            You must be strictly BIASED towards finding reasons to deduct points. Act as a harsh professor grading a student's final paper.

            **INPUT CONTEXT:**
            - Original Transcript: {transcription}
            - Generated Summary: {summary}
            
            --- STICKT 10-POINT DEDUCTION FRAMEWORK ---
            
            Your final score is 10 minus all deductions. You must justify every deduction made in each category.

            I. Faithfulness & Consistency (Target: 3 Points) - [The Lie Detector]
               - Primary Check: Does every claim align with the {transcription} context?
               - Deduction Rule: -1.0 point for every factual error, date mistake, or significant misquote (Hallucination).

            II. Structure & Relevance (Target: 3 Points) - [The Architect]
               - Primary Check: Is the summary correctly categorized (Academic/Procedural/General) and does it follow the correct structure for that type? Is the information relevant to the video's main topic?
               - Deduction Rule: -1.0 point for wrong classification. -0.5 point for missing required structural headers (e.g., missing 'Key Definitions' in an academic report).

            III. Completeness & Density (Target: 2 Points) - [The Detail Hunter]
               - Primary Check: Does the report capture ALL critical arguments, definitions, or procedural steps?
               - Deduction Rule: -1.0 point for omitting a major section or a required step in a procedural video. -0.5 point for unnecessary padding or repetition.

            IV. Fluency & Tone (Target: 2 Points) - [The Editor]
               - Primary Check: Is the language clear, professional, and free of grammar errors?
               - Deduction Rule: -0.5 point for each severe grammar error or non-fluent sentence construction.
            
            ---
            
            **OUTPUT REQUIREMENT (CoT Mandatory):**
            You must output a JSON object containing your step-by-step reasoning and the final score. 
            Do NOT write any prose or explanation outside of the JSON structure.

            **JSON Structure:**
            {
                "audit_critique": "Detailed list of every flaw found in each category (I, II, III, IV).",
                "total_deductions": <FLOAT_NUMBER>,
                "final_score": <INTEGER_BETWEEN_1_AND_10>
            }
            """,
            expected_output="A single JSON object containing the 'audit_critique', 'total_deductions', and 'final_score' (as an integer).",
            agent=self.evaluator()
        )
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
    
    # [新增] 创建审查 Crew
    def create_evaluation_crew(self) -> Crew:
        return Crew(
            agents=[self.evaluator()],
            tasks=[self.evaluation_task()],
            process=Process.sequential,
            verbose=True
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