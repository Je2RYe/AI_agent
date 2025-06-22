# VideoSummary Crew

Welcome to the VideoSummary Crew project, powered by [crewAI](https://crewai.com). This template is designed to help you set up a multi-agent AI system with ease, leveraging the powerful and flexible framework provided by crewAI. Our goal is to enable your agents to collaborate effectively on complex tasks, maximizing their collective intelligence and capabilities.

## Installation

Ensure you have Python >=3.10 <3.14 installed on your system. This project uses [UV](https://docs.astral.sh/uv/) for dependency management and package handling, offering a seamless setup and execution experience.

First, if you haven't already, install uv:

```bash
pip install uv
```

Next, navigate to your project directory and install the dependencies:

(Optional) Lock the dependencies and install them by using the CLI command:
```bash
crewai install
```
### Customizing

**Add your `OPENAI_API_KEY` into the `.env` file**

- Modify `src/video_summary/config/agents.yaml` to define your agents
- Modify `src/video_summary/config/tasks.yaml` to define your tasks
- Modify `src/video_summary/crew.py` to add your own logic, tools and specific args
- Modify `src/video_summary/main.py` to add custom inputs for your agents and tasks

## Running the Project

To kickstart your crew of AI agents and begin task execution, run this from the root folder of your project:

```bash
$ crewai run
```

This command initializes the video-summary Crew, assembling the agents and assigning them tasks as defined in your configuration.

This example, unmodified, will run the create a `report.md` file with the output of a research on LLMs in the root folder.

## Understanding Your Crew# Video Summary - Portable Setup Guide

This guide explains how to set up the Video Summary application on any device, making it portable and environment-independent.

## 🚀 Quick Start

### 1. Run the Setup Script
```bash
python video_summary/setup_environment.py
```

This script will:
- ✅ Check Python version compatibility
- 🎬 Verify FFmpeg installation
- 📦 Check required Python packages
- 🚀 Detect GPU availability
- ⚙️ Create environment configuration file

### 2. Install Dependencies
```bash
pip install -r requirement.txt
```

### 3. Configure Environment
Edit the `.env` file created by the setup script:
```env
# Required: Your OpenAI API key
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Custom FFmpeg path (if auto-detection fails)
# FFMPEG_PATH=/path/to/ffmpeg/bin

# Optional: Force CPU usage
# FORCE_CPU=true

# Optional: Disable GPU
# ENABLE_GPU=false
```

### 4. Run the Application
```bash
streamlit run video_summary/src/video_summary/main.py
```

## 🔧 FFmpeg Installation

The application automatically detects FFmpeg in these locations:

### Windows
- `C:/ffmpeg/bin/` (common installation)
- System PATH
- Environment variable `FFMPEG_PATH`

### macOS
- `/usr/local/bin/` (Homebrew)
- `/opt/homebrew/bin/` (Apple Silicon Homebrew)
- System PATH

### Linux
- `/usr/bin/` (package manager)
- `/usr/local/bin/` (manual installation)
- System PATH

### Manual Installation
If auto-detection fails, install FFmpeg:

**Windows:**
1. Download from https://ffmpeg.org/download.html
2. Extract to `C:\ffmpeg\`
3. Add `C:\ffmpeg\bin` to PATH or set `FFMPEG_PATH=C:\ffmpeg\bin`

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

## 🎯 Configuration Options

### Whisper Model Selection
Edit `video_summary/src/video_summary/config/settings.py`:
```python
class AppConfig:
    WHISPER_MODEL = "small"  # Options: tiny, base, small, medium, large
```

**Model Comparison:**
- `tiny`: Fastest, lowest accuracy
- `base`: Good balance
- `small`: Recommended (default)
- `medium`: Better accuracy, slower
- `large`: Best accuracy, slowest

### GPU Configuration
The application automatically detects and uses:
- **NVIDIA GPUs**: CUDA acceleration
- **Apple Silicon**: MPS acceleration  
- **CPU**: Fallback option

**Force CPU usage:**
```python
class AppConfig:
    FORCE_CPU = True
```

**Disable GPU:**
```python
class AppConfig:
    ENABLE_GPU = False
```

## 🌍 Cross-Platform Compatibility

### Supported Operating Systems
- ✅ Windows 10/11
- ✅ macOS 10.15+
- ✅ Linux (Ubuntu, Debian, CentOS)

### Supported Audio Formats
- MP3, WAV, M4A, FLAC, OGG

### File Size Limits
- Default: 100MB maximum
- Configurable in `settings.py`

## 🔍 Troubleshooting

### FFmpeg Not Found
1. Run the setup script: `python video_summary/setup_environment.py`
2. Install FFmpeg manually
3. Set `FFMPEG_PATH` environment variable

### GPU Not Working
1. Check GPU detection: `python video_summary/test_gpu.py`
2. Install CUDA drivers (NVIDIA)
3. Install PyTorch with CUDA support

### Missing Dependencies
```bash
pip install torch torchaudio
pip install -r requirement.txt
```

### Permission Errors
- **Windows**: Run as Administrator
- **Linux/macOS**: Use `sudo` if needed

## 📁 Project Structure
```
video_summary/
├── src/video_summary/
│   ├── config/
│   │   ├── settings.py      # Configuration system
│   │   ├── agents.yaml      # Agent definitions
│   │   └── tasks.yaml       # Task definitions
│   ├── crew.py              # Main crew logic
│   └── main.py              # Streamlit interface
├── setup_environment.py     # Environment setup
├── test_gpu.py             # GPU testing
└── README_PORTABLE.md      # This file
```

## 🚀 Performance Tips

### For Best Performance:
1. **Use GPU**: Enable CUDA/MPS for 5-10x speed improvement
2. **Model Selection**: Use 'small' for speed, 'large' for accuracy
3. **Close Applications**: Free up GPU memory
4. **File Size**: Keep audio files under 50MB for faster processing

### For CPU-Only Systems:
1. Use 'tiny' or 'base' Whisper models
2. Process shorter audio files
3. Close unnecessary applications

## 🔄 Updates and Maintenance

### Updating Dependencies
```bash
pip install --upgrade -r requirement.txt
```

### Checking for Updates
```bash
python video_summary/setup_environment.py
```

### Backup Configuration
Save your `.env` file and `settings.py` modifications for easy restoration.

## 📞 Support

If you encounter issues:
1. Run the setup script for diagnostics
2. Check the troubleshooting section
3. Verify all dependencies are installed
4. Ensure FFmpeg is properly configured

---

**Happy Transcribing! 🎵➡️📝** 

The video-summary Crew is composed of multiple AI agents, each with unique roles, goals, and tools. These agents collaborate on a series of tasks, defined in `config/tasks.yaml`, leveraging their collective skills to achieve complex objectives. The `config/agents.yaml` file outlines the capabilities and configurations of each agent in your crew.

## Support

For support, questions, or feedback regarding the VideoSummary Crew or crewAI.
- Visit our [documentation](https://docs.crewai.com)
- Reach out to us through our [GitHub repository](https://github.com/joaomdmoura/crewai)
- [Join our Discord](https://discord.com/invite/X4JWnZnxPb)
- [Chat with our docs](https://chatg.pt/DWjSBZn)

Let's create wonders together with the power and simplicity of crewAI.
