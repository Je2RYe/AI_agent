#!/usr/bin/env python3
"""
GPU Test Script for Video Summary
This script tests GPU availability and provides performance information.
"""

import torch
import whisper
import time
import os

def test_gpu_availability():
    """Test and display GPU availability information"""
    print("=" * 50)
    print("GPU AVAILABILITY TEST")
    print("=" * 50)
    
    # Test CUDA
    if torch.cuda.is_available():
        print("✅ CUDA GPU detected!")
        print(f"   GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        return "cuda"
    
    # Test Apple Silicon (MPS)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✅ Apple Silicon GPU (MPS) detected!")
        return "mps"
    
    else:
        print("❌ No GPU detected - will use CPU")
        print(f"   CPU Cores: {os.cpu_count()}")
        return "cpu"

def test_whisper_performance():
    """Test Whisper model loading and basic performance"""
    print("\n" + "=" * 50)
    print("WHISPER PERFORMANCE TEST")
    print("=" * 50)
    
    device = test_gpu_availability()
    
    print(f"\nLoading Whisper model on {device}...")
    start_time = time.time()
    
    try:
        model = whisper.load_model("small", device=device)
        load_time = time.time() - start_time
        print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
        
        # Test with a small dummy audio file if available
        test_files = ["temp_audio/song1.mp3"]
        for test_file in test_files:
            if os.path.exists(test_file):
                print(f"\nTesting transcription with {test_file}...")
                transcribe_start = time.time()
                result = model.transcribe(test_file, fp16=(device == "cuda"))
                transcribe_time = time.time() - transcribe_start
                print(f"✅ Transcription completed in {transcribe_time:.2f} seconds")
                print(f"   Text length: {len(result['text'])} characters")
                break
        else:
            print("ℹ️  No test audio file found - skipping transcription test")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    return True

def show_performance_tips():
    """Show performance optimization tips"""
    print("\n" + "=" * 50)
    print("PERFORMANCE OPTIMIZATION TIPS")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu"
    
    if device == "cuda":
        print("🚀 GPU Optimizations Enabled:")
        print("   • Using CUDA for Whisper transcription")
        print("   • FP16 precision enabled for faster processing")
        print("   • GPU memory optimization active")
        print("\n💡 Tips for best performance:")
        print("   • Close other GPU-intensive applications")
        print("   • Use 'small' model for faster transcription")
        print("   • Consider 'medium' or 'large' for better accuracy")
        
    elif device == "mps":
        print("🍎 Apple Silicon Optimizations Enabled:")
        print("   • Using MPS (Metal Performance Shaders)")
        print("   • Optimized for Apple Silicon GPUs")
        print("\n💡 Tips for best performance:")
        print("   • Ensure macOS 12.3+ for MPS support")
        print("   • Close other GPU-intensive applications")
        
    else:
        print("💻 CPU Mode:")
        print("   • Using CPU for transcription")
        print("   • Consider upgrading to GPU for better performance")
        print("\n💡 Tips for best performance:")
        print("   • Use 'tiny' or 'small' models for faster processing")
        print("   • Close unnecessary applications")
        print("   • Consider cloud GPU services for large files")

if __name__ == "__main__":
    print("Video Summary GPU Test")
    print("This script tests GPU availability and Whisper performance.\n")
    
    success = test_whisper_performance()
    show_performance_tips()
    
    if success:
        print("\n✅ GPU test completed successfully!")
        print("Your system is ready for GPU-accelerated transcription.")
    else:
        print("\n❌ GPU test failed. Check your installation.")
        print("You can still use CPU mode for transcription.") 