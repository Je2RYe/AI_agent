#!/usr/bin/env python
import sys
import warnings
from dotenv import load_dotenv
import streamlit as st
from crewai import Crew
import os

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crew import VideoSummary

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information
load_dotenv()

def main():
    st.title("Content Summarizer")

    # Sidebar for inputs
    st.sidebar.header("Content Summarizer")
    youtube_url = st.sidebar.text_input("Enter YouTube URL:")
    
    if st.sidebar.button("Summarize Video from URL"):
        if youtube_url:
            with st.spinner("Summarizing video..."):
                inputs = {'content': youtube_url}
                video_summary_crew = VideoSummary()
                video_summary_crew.crew().kickoff(inputs=inputs)
                
                st.subheader("Generated Summary:")
                
                summary_file_path = "Video_Summary.txt"
                if os.path.exists(summary_file_path):
                    with open(summary_file_path, "r", encoding="utf-8") as f:
                        summary_content = f.read()
                        st.markdown(summary_content)
                else:
                    st.warning("Summary file 'Video_Summary.txt' not found.")
        else:
            st.sidebar.warning("Please enter a YouTube URL.")

    st.sidebar.markdown("---")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload an audio file (MP3, WAV, M4A)",
        type=["mp3", "wav", "m4a"]
    )
    
    if st.sidebar.button("Summarize from File"):
        if uploaded_file is not None:
            with st.spinner("Summarizing from file..."):
                # Save the uploaded file temporarily
                temp_dir = "temp_audio"
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                inputs = {'content': file_path}
                video_summary_crew = VideoSummary()
                video_summary_crew.crew().kickoff(inputs=inputs)

                st.subheader("Generated Summary:")
                
                summary_file_path = "Video_Summary.txt"
                if os.path.exists(summary_file_path):
                    with open(summary_file_path, "r", encoding="utf-8") as f:
                        summary_content = f.read()
                        st.markdown(summary_content)
                else:
                    st.warning("Summary file 'Video_Summary.txt' not found.")

                # Clean up the temporary file
                os.remove(file_path)
        else:
            st.sidebar.warning("Please upload an audio file.")

if __name__ == "__main__":
    main()
    
