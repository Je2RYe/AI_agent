#!/usr/bin/env python
import sys
import warnings
from dotenv import load_dotenv
import streamlit as st
from crewai import Crew
import os
import time
import threading

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crew import VideoSummary

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information
load_dotenv()


def run_summarization(inputs, done_flag):
    video_summary_crew = VideoSummary()
    video_summary_crew.crew().kickoff(inputs=inputs)
    done_flag['finished'] = True


def show_progress_placebo(inputs):
    st.title("🚀 Progress Bar")
    progress_text = st.empty()
    progress_bar = st.progress(0)

    done_flag = {'finished': False}
    thread = threading.Thread(target=run_summarization, args=(inputs, done_flag))
    thread.start()

    progress = 0.0

    # Detect theme
    dark_mode_active = st.get_option("theme.base") == "dark"
    text_color = "#ffffff" if dark_mode_active else "#000000"
    success_color = "#66ff66" if dark_mode_active else "#006400"

    # Slowly increase progress up to 99%
    while progress < 0.99 and not done_flag['finished']:
        progress += 0.01
        progress_bar.progress(progress)
        progress_text.markdown(
            f"<p style='color: {text_color}; font-weight: bold;'>Progress: {int(progress * 100)}%</p>",
            unsafe_allow_html=True
        )
        time.sleep(0.1)

    # Wait for the real task to finish if it hasn't yet
    while not done_flag['finished']:
        time.sleep(0.1)

    # Jump to 100% when done
    progress_bar.progress(1.0)
    progress_text.markdown(
        f"<p style='color: {text_color}; font-weight: bold;'>Progress: 100%</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='color: {success_color}; font-weight: bold;'>🎉 Task Completed!</p>",
        unsafe_allow_html=True
    )
    thread.join()



def set_theme(dark_mode: bool):
    if dark_mode:
        sidebar_bg = "#1f2430"      # Darker, cooler sidebar
        main_bg = "#121521"         # Dark but different shade for main area
        text_color = "#e0e6f1"      # Soft off-white text
        input_bg = "#232a3a"
        input_text = "#e0e6f1"
        button_bg = "#2a3245"
        button_text = "#f0f3ff"
        button_hover_bg = "#3b4660"
    else:
        sidebar_bg = "#e6e8eb"      # Light gray sidebar, distinct from white main
        main_bg = "#ffffff"         # Pure white main area
        text_color = "#1c1c1c"      # Dark charcoal text
        input_bg = "#f7f8fa"
        input_text = "#1c1c1c"
        button_bg = "#f0f0f3"
        button_text = "#333333"
        button_hover_bg = "#d6d6d9"

    st.markdown(
        f"""
        <style>
        /* Font family for entire app */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
            font-family: "Inter", "Segoe UI", "Helvetica Neue", sans-serif;
            background-color: {main_bg} !important;
            color: {text_color} !important;
        }}

        /* Sidebar styling */
        section[data-testid="stSidebar"] {{
            background-color: {sidebar_bg} !important;
            color: {text_color} !important;
            box-shadow: 2px 0 8px rgba(0, 0, 0, 0.15);
        }}
        section[data-testid="stSidebar"] * {{
            color: {text_color} !important;
        }}

        /* Main content background and text */
        .main, .main * {{
            background-color: {main_bg} !important;
            color: {text_color} !important;
        }}

        /* Inputs */
        input, textarea {{
            background-color: {input_bg} !important;
            color: {input_text} !important;
            border: 1px solid #999999 !important;
            border-radius: 6px !important;
            font-weight: 500;
            font-size: 0.95rem;
            padding: 0.4rem 0.7rem;
        }}

        /* File uploader box */
        [data-testid="stFileUploader"] > div > div {{
            background-color: {input_bg} !important;
            color: {input_text} !important;
            border: 1px solid #999999 !important;
            border-radius: 6px !important;
        }}
        [data-testid="stFileUploader"] > div > div * {{
            color: {input_text} !important;
        }}

        /* File uploader drag-and-drop area */
        [data-testid="stFileUploaderDropzone"] {{
            background-color: {input_bg} !important;
            color: {input_text} !important;
            border: 1px solid #999999 !important;
            border-radius: 6px !important;
        }}
        [data-testid="stFileUploaderDropzone"] * {{
            color: {input_text} !important;
        }}

        /* Buttons */
        button {{
            background-color: {button_bg} !important;
            color: {button_text} !important;
            border: none !important;
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            font-weight: 600;
            font-size: 1rem;
            box-shadow: 0 3px 8px rgba(0,0,0,0.1);
            transition: background-color 0.3s ease, box-shadow 0.3s ease;
        }}
        button:hover {{
            background-color: {button_hover_bg} !important;
            box-shadow: 0 6px 15px rgba(0,0,0,0.2);
            color: {button_text} !important;
            cursor: pointer;
        }}

        /* Labels and placeholders */
        label, ::placeholder {{
            color: {text_color} !important;
            font-weight: 500;
            font-size: 0.95rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )



def main():
    dark_mode = st.sidebar.checkbox("Dark Mode", value=False)
    set_theme(dark_mode)

    st.title("Content Summarizer")

    # Sidebar for inputs
    st.sidebar.header("Content Summarizer")
    youtube_url = st.sidebar.text_input("Enter YouTube URL:")

    if st.sidebar.button("Summarize Video from URL"):
        if youtube_url:
            ##with st.spinner("Summarizing video..."):

            inputs = {'content': youtube_url}
            show_progress_placebo(inputs)
            ##video_summary_crew = VideoSummary()
            ##video_summary_crew.crew().kickoff(inputs=inputs)

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
            ##with st.spinner("Summarizing from file..."):
            # Save the uploaded file temporarily

            temp_dir = "temp_audio"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            inputs = {'content': file_path}
            show_progress_placebo(inputs)
            ##video_summary_crew = VideoSummary()
            ##video_summary_crew.crew().kickoff(inputs=inputs)

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
