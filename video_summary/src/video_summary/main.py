#!/usr/bin/env python
import sys
import warnings
from dotenv import load_dotenv
import streamlit as st
from crewai import Crew
import os
import time
import threading

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from crew import VideoSummary

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

load_dotenv()


def run_summarization(inputs, done_flag):
    video_summary_crew = VideoSummary()
    video_summary_crew.create_summarization_crew().kickoff(inputs=inputs)
    done_flag['finished'] = True


def show_progress_placebo(inputs, dark_mode):
    st.title("🚀 Generating Summary")
    progress_text = st.empty()
    progress_bar = st.progress(0)

    done_flag = {'finished': False}
    thread = threading.Thread(target=run_summarization, args=(inputs, done_flag))
    thread.start()

    progress = 0.0
    text_color = "#ffffff" if dark_mode else "#000000"
    success_color = "#66ff66" if dark_mode else "#006400"

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
    if 'messages' in st.session_state:
        del st.session_state['messages']


def set_theme(dark_mode: bool):
    if dark_mode:
        sidebar_bg, main_bg, text_color, input_bg, input_text, button_bg, button_text, button_hover_bg, chat_bg = (
            "#1f2430", "#121521", "#e0e6f1", "#232a3a", "#e0e6f1", "#2a3245", "#f0f3ff", "#3b4660", "#2a2f3d"
        )
    else:
        sidebar_bg, main_bg, text_color, input_bg, input_text, button_bg, button_text, button_hover_bg, chat_bg = (
            "#e6e8eb", "#ffffff", "#1c1c1c", "#f7f8fa", "#1c1c1c", "#f0f0f3", "#333333", "#d6d6d9", "#f2f2f2"
        )

    st.markdown(f"""
        <style>
        /* === Global Background and Text === */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"]  {{
            background-color: {main_bg} !important;
            color: {text_color} !important;
        }}

        /* === Sidebar Styling === */
        section[data-testid="stSidebar"] {{
            background-color: {sidebar_bg} !important;
            box-shadow: 2px 0 8px rgba(0, 0, 0, 0.15);
        }}
        section[data-testid="stSidebar"] * {{
            color: {text_color} !important;
        }}

        /* === Selectbox (Dropdown) === */
        .stSelectbox div[data-baseweb="select"] > div:first-child {{
            background-color: {button_bg} !important;
            color: {button_text} !important;
            border: 1px solid #999 !important;
            border-radius: 6px !important;
         
        }}

        /* === Main Content Area === */
        .main, .main * {{
            background-color: {main_bg} !important;
            color: {text_color} !important;
           
        }}

        [data-testid="stTextInput"] input {{
            background-color: {input_bg} !important;
            color: {input_text} !important;
            padding: 0.5rem !important;
            font-size: 1rem !important;
            border: 1px solid #999 !important;
            border-radius: 6px !important;
        }}

        /* === File Uploader === */
        [data-testid="stFileUploader"] > div > div,
        [data-testid="stFileUploaderDropzone"] {{
            background-color: {input_bg} !important;
            color: {input_text} !important;
            border: 1px solid #999 !important;
            border-radius: 6px !important;
        }}


        /* === Buttons === */
        button {{
            background-color: {button_bg} !important;
            color: {button_text} !important;
            border: none !important;
            padding: 0.6rem 1.2rem;
            border-radius: 8px !important;
            font-weight: 600;
            box-shadow: 0 3px 8px rgba(0,0,0,0.1);
            transition: background-color 0.3s ease, box-shadow 0.3s ease;
        }}
        button:hover {{
            background-color: {button_hover_bg} !important;
            box-shadow: 0 6px 15px rgba(0,0,0,0.2);
            cursor: pointer;
        }}

        /* === Chat Messages === */
        [data-testid="stChatMessage"] {{
            background-color: {chat_bg} !important;
            border-radius: 12px !important;
            padding: 1rem;
            margin-bottom: 1rem;
        }}
        [data-testid="stChatMessageContent"] {{
            color: {text_color} !important;
        }}

        /* === Optional: Make chat input blend better === */
        div[data-testid="stChatInput"] {{
            border-top: 1px solid rgba(255,255,255,0.05);
            color: {text_color} !important;
        }}
        div[data-testid="stChatInput"] * {{
            background-color: {input_bg} !important;
            color: {text_color} !important;
            box-shadow: none !important;
        }}
        div[data-testid="stChatInput"] textarea {{
            background-color: {input_bg} !important;
            color: {text_color} !important;
  
        }}
        div[data-testid="stChatInput"] button {{
            background-color: {button_bg} !important;
            color: {button_text} !important;
            border: 1px solid #666 !important;
            border-radius: 0 12px 12px 0 !important;
        }}
        </style>
    """, unsafe_allow_html=True)



def main():
    # --- Language Selection ---
    language = st.sidebar.selectbox("🌐 Language / Langue", ["English", "Français"])

    translations = {
        "English": {
            "dark_mode": "Dark Mode",
            "title": "Content Summarizer",
            "youtube_input": "Enter YouTube URL:",
            "summarize_url": "Summarize Video from URL",
            "upload_file": "Upload an audio file (MP3, WAV, M4A)",
            "summarize_file": "Summarize from File",
            "warning_url": "Please enter a YouTube URL.",
            "warning_file": "Please upload an audio file.",
            "summary_title": "Generated Summary",
            "chat_title": "Chat About The Summary",
            "chat_input": "Ask a question about the summary...",
            "summary_info": "Your generated summary and chat will appear here once you provide a URL or file."
        },
        "Français": {
            "dark_mode": "Mode Sombre",
            "title": "Résumeur de Contenu",
            "youtube_input": "Entrez l'URL YouTube :",
            "summarize_url": "Résumer la vidéo à partir de l'URL",
            "upload_file": "Téléversez un fichier audio (MP3, WAV, M4A)",
            "summarize_file": "Résumer à partir du fichier",
            "warning_url": "Veuillez entrer une URL YouTube.",
            "warning_file": "Veuillez téléverser un fichier audio.",
            "summary_title": "Résumé Généré",
            "chat_title": "Discuter du Résumé",
            "chat_input": "Posez une question sur le résumé...",
            "summary_info": "Votre résumé généré et le chat apparaîtront ici une fois que vous aurez fourni une URL ou un fichier."
        }
    }

    t = translations[language]
    # Persist dark mode toggle across reruns
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = True
    st.session_state.dark_mode = st.sidebar.checkbox(t["dark_mode"], value=st.session_state.dark_mode)
    dark_mode = st.session_state.dark_mode

    # Set theme based on stored dark mode state
    set_theme(dark_mode)

    st.sidebar.title(t["title"])

    youtube_url = st.sidebar.text_input(t["youtube_input"])
    if st.sidebar.button(t["summarize_url"]):
        if youtube_url:
            inputs = {'content': youtube_url}
            show_progress_placebo(inputs, dark_mode)
        else:
            st.sidebar.warning(t["warning_url"])

    st.sidebar.markdown("---")

    uploaded_file = st.sidebar.file_uploader(t["upload_file"], type=["mp3", "wav", "m4a"])
    if st.sidebar.button(t["summarize_file"]):
        if uploaded_file is not None:
            temp_dir = "temp_audio"
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            inputs = {'content': file_path}
            show_progress_placebo(inputs, dark_mode)
            os.remove(file_path)
        else:
            st.sidebar.warning(t["warning_file"])

    st.title("📄 " + t["summary_title"])
    summary_file_path = "Video_Summary.txt"
    if os.path.exists(summary_file_path):
        with open(summary_file_path, "r", encoding="utf-8") as f:
            summary_content = f.read()

        st.subheader(t["summary_title"])
        st.markdown(summary_content)
        st.markdown("---")
        st.subheader(t["chat_title"])

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input(t["chat_input"]):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    video_summary_crew = VideoSummary()
                    inputs = {
                        'summary': summary_content,
                        'user_message': prompt
                    }
                    chat_crew = video_summary_crew.create_chat_crew()
                    response = chat_crew.kickoff(inputs=inputs)
                    st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info(t["summary_info"])


if __name__ == "__main__":
    main()
