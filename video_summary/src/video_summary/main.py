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
    """Kick off the summarization crew in a separate thread."""
    video_summary_crew = VideoSummary()
    # Use the specific method for the summarization crew
    video_summary_crew.create_summarization_crew().kickoff(inputs=inputs)
    done_flag['finished'] = True


def show_progress_placebo(inputs, dark_mode):
    """Display a placebo progress bar while the crew is running."""
    st.title("🚀 Generating Summary")
    progress_text = st.empty()
    progress_bar = st.progress(0)

    done_flag = {'finished': False}
    thread = threading.Thread(target=run_summarization, args=(inputs, done_flag))
    thread.start()

    progress = 0.0

    dark_mode_active = dark_mode
    text_color = "#ffffff" if dark_mode_active else "#000000"
    success_color = "#66ff66" if dark_mode_active else "#006400"

    while progress < 0.99 and not done_flag['finished']:
        progress += 0.01
        progress_bar.progress(progress)
        progress_text.markdown(
            f"<p style='color: {text_color}; font-weight: bold;'>Progress: {int(progress * 100)}%</p>",
            unsafe_allow_html=True
        )
        time.sleep(0.1)

    while not done_flag['finished']:
        time.sleep(0.1)

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
    # Clear session chat history for the new summary
    if 'messages' in st.session_state:
        del st.session_state['messages']


def set_theme(dark_mode: bool):
    """Set the Streamlit theme with custom colors."""
    if dark_mode:
        sidebar_bg, main_bg, text_color, input_bg, input_text, button_bg, button_text, button_hover_bg = ("#1f2430", "#121521", "#e0e6f1", "#232a3a", "#e0e6f1", "#2a3245", "#f0f3ff", "#3b4660")
    else:
        sidebar_bg, main_bg, text_color, input_bg, input_text, button_bg, button_text, button_hover_bg = ("#e6e8eb", "#ffffff", "#1c1c1c", "#f7f8fa", "#1c1c1c", "#f0f0f3", "#333333", "#d6d6d9")

    st.markdown(f"""
        <style>
        /* General styling */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
            font-family: "Inter", "Segoe UI", "Helvetica Neue", sans-serif;
            background-color: {main_bg} !important;
            color: {text_color} !important;
        }}
        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: {sidebar_bg} !important;
            box-shadow: 2px 0 8px rgba(0, 0, 0, 0.15);
        }}
        section[data-testid="stSidebar"] * {{ color: {text_color} !important; }}
        /* Main content */
        .main, .main * {{
            background-color: {main_bg} !important;
            color: {text_color} !important;
        }}
        /* Inputs and Buttons */
        input, textarea {{
            background-color: {input_bg} !important;
            color: {input_text} !important; border: 1px solid #999999 !important; border-radius: 6px !important;
        }}
        [data-testid="stFileUploader"] > div > div, [data-testid="stFileUploaderDropzone"] {{
            background-color: {input_bg} !important; border: 1px solid #999999 !important; border-radius: 6px !important;
        }}
        [data-testid="stFileUploader"] * {{ color: {input_text} !important; }}
        button {{
            background-color: {button_bg} !important; color: {button_text} !important;
            border: none !important; padding: 0.6rem 1.2rem; border-radius: 8px;
            font-weight: 600; box-shadow: 0 3px 8px rgba(0,0,0,0.1);
            transition: background-color 0.3s ease, box-shadow 0.3s ease;
        }}
        button:hover {{
            background-color: {button_hover_bg} !important;
            box-shadow: 0 6px 15px rgba(0,0,0,0.2); cursor: pointer;
        }}
        </style>
    """, unsafe_allow_html=True)


def main():
    """Main function to run the Streamlit app."""
    dark_mode = st.sidebar.checkbox("Dark Mode", value=True)
    set_theme(dark_mode)

    st.sidebar.title("Content Summarizer")
    
    # --- URL Input ---
    youtube_url = st.sidebar.text_input("Enter YouTube URL:")
    if st.sidebar.button("Summarize Video from URL"):
        if youtube_url:
            inputs = {'content': youtube_url}
            show_progress_placebo(inputs, dark_mode)
        else:
            st.sidebar.warning("Please enter a YouTube URL.")

    st.sidebar.markdown("---")

    # --- File Uploader Input ---
    uploaded_file = st.sidebar.file_uploader(
        "Upload an audio file (MP3, WAV, M4A)", type=["mp3", "wav", "m4a"]
    )
    if st.sidebar.button("Summarize from File"):
        if uploaded_file is not None:
            temp_dir = "temp_audio"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            inputs = {'content': file_path}
            show_progress_placebo(inputs, dark_mode)
            os.remove(file_path) # Clean up
        else:
            st.sidebar.warning("Please upload an audio file.")

    # --- Main Display Area for Summary and Chat ---
    st.title("📄 Summary and Chat")
    summary_file_path = "Video_Summary.txt"
    if os.path.exists(summary_file_path):
        with open(summary_file_path, "r", encoding="utf-8") as f:
            summary_content = f.read()
        
        st.subheader("Generated Summary")
        st.markdown(summary_content)
        st.markdown("---")
        st.subheader("Chat About The Summary")

        # Initialize or display chat messages
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input field
        if prompt := st.chat_input("Ask a question about the summary..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get response from chat agent
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    video_summary_crew = VideoSummary()
                    inputs = {
                        'summary': summary_content,
                        'user_message': prompt
                    }
                    # Use the specific method for the chat crew
                    chat_crew = video_summary_crew.create_chat_crew()
                    response = chat_crew.kickoff(inputs=inputs)
                    st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("Your generated summary and chat will appear here once you provide a URL or file.")

if __name__ == "__main__":
    main()