#!/usr/bin/env python3
"""
Environment Setup Script for Video Summary
This script helps configure the environment and check for required dependencies.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ is required.")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True

def check_ffmpeg():
    """Check FFmpeg installation"""
    print("\n🎬 Checking FFmpeg installation...")
    
    # Try to run ffmpeg
    try:
        result = subprocess.run(["ffmpeg", "-version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ FFmpeg found in system PATH")
            return True
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Check local project FFmpeg installation
    current_dir = Path(__file__).parent
    local_ffmpeg_paths = [
        current_dir / "ffmpeg" / "bin" / "ffmpeg.exe",  # Windows
        current_dir / "ffmpeg" / "bin" / "ffmpeg",      # Linux/macOS
        current_dir / "ffmpeg" / "ffmpeg.exe",          # Direct in ffmpeg folder (Windows)
        current_dir / "ffmpeg" / "ffmpeg",              # Direct in ffmpeg folder (Linux/macOS)
    ]
    
    for path in local_ffmpeg_paths:
        if path.exists():
            print(f"✅ FFmpeg found in project directory: {path}")
            # Add to PATH for this session
            ffmpeg_bin_dir = str(path.parent)
            if ffmpeg_bin_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = ffmpeg_bin_dir + os.pathsep + os.environ.get("PATH", "")
                print(f"   Added {ffmpeg_bin_dir} to PATH")
            return True
    
    # Check common installation paths
    common_paths = [
        "C:/ffmpeg/bin/ffmpeg.exe",
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/opt/homebrew/bin/ffmpeg"
    ]
    
    for path in common_paths:
        if Path(path).exists():
            print(f"✅ FFmpeg found at: {path}")
            return True
    
    print("❌ FFmpeg not found. Please install FFmpeg:")
    print("   Windows: Download from https://ffmpeg.org/download.html")
    print("   macOS: brew install ffmpeg")
    print("   Linux: sudo apt install ffmpeg (Ubuntu/Debian)")
    print("   Or set FFMPEG_PATH environment variable to point to ffmpeg installation")
    return False

def check_python_packages():
    """Check required Python packages"""
    print("\n📦 Checking Python packages...")
    
    required_packages = [
        "torch",
        "whisper",
        "yt-dlp",
        "youtube-transcript-api",
        "crewai",
        "streamlit"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📋 Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_gpu():
    """Check GPU availability"""
    print("\n🚀 Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ CUDA GPU detected: {gpu_count} device(s)")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            return True
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✅ Apple Silicon GPU (MPS) detected")
            return True
        else:
            print("ℹ️  No GPU detected - will use CPU")
            return True
    except ImportError:
        print("❌ PyTorch not installed - cannot check GPU")
        return False

def create_env_file():
    """Create .env file with configuration"""
    print("\n⚙️  Creating environment configuration...")
    
    env_content = """# Video Summary Environment Configuration

# FFmpeg path (optional - will auto-detect if not set)
# FFMPEG_PATH=/path/to/ffmpeg/bin

# OpenAI API key (required for CrewAI)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Force CPU usage even if GPU is available
# FORCE_CPU=true

# Optional: Disable GPU usage
# ENABLE_GPU=false
"""
    
    env_file = Path(".env")
    if env_file.exists():
        print("ℹ️  .env file already exists")
    else:
        with open(env_file, "w") as f:
            f.write(env_content)
        print("✅ Created .env file")
        print("   Please edit .env file and add your OpenAI API key")

def main():
    """Main setup function"""
    print("🎯 Video Summary Environment Setup")
    print("=" * 50)
    
    checks = [
        check_python_version(),
        check_ffmpeg(),
        check_python_packages(),
        check_gpu()
    ]
    
    create_env_file()
    
    print("\n" + "=" * 50)
    if all(checks):
        print("✅ All checks passed! Your environment is ready.")
        print("\n🚀 To run the application:")
        print("   streamlit run video_summary/src/video_summary/main.py")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("\n💡 Tips:")
        print("   - Install missing packages: pip install -r requirement.txt")
        print("   - Install FFmpeg from https://ffmpeg.org/download.html")
        print("   - Set up your OpenAI API key in the .env file")

if __name__ == "__main__":
    main() 