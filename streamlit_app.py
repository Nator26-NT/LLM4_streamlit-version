import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
import config
from models import CVModels
from checker import EligibilityChecker

# Page config
st.set_page_config(page_title="Exam Eligibility Checker", page_icon="📝", layout="wide")

# Initialize CV components (cached)
@st.cache_resource
def init_models():
    models = CVModels(config)
    checker = EligibilityChecker(config, models)
    return models, checker

models, checker = init_models()

# Theme toggle in session state
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

# Custom CSS based on theme
def get_css():
    if st.session_state.theme == "dark":
        return """
        <style>
        .stApp {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #ffffff;
        }
        .card {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 1.5rem;
            text-align: center;
            transition: transform 0.3s, box-shadow 0.3s;
            box-shadow: 0 8px 20px rgba(0,0,0,0.2);
            margin: 1rem;
            height: 100%;
        }
        .card:hover {
            transform: translateY(-10px);
            box-shadow: 0 15px 30px rgba(0,0,0,0.3);
        }
        h1, h2, h3 {
            color: #ffffff;
        }
        .stButton button {
            background: #00b4db;
            color: white;
            border-radius: 30px;
            border: none;
            padding: 0.5rem 1.5rem;
            transition: 0.3s;
        }
        .stButton button:hover {
            background: #0083b0;
            transform: scale(1.02);
        }
        .stFileUploader {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 0.5rem;
        }
        .success-text {
            color: #00ffaa;
            font-weight: bold;
        }
        .error-text {
            color: #ff5555;
            font-weight: bold;
        }
        </style>
        """
    else:
        return """
        <style>
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            color: #1a1a2e;
        }
        .card {
            background: rgba(255,255,255,0.9);
            border-radius: 20px;
            padding: 1.5rem;
            text-align: center;
            transition: transform 0.3s, box-shadow 0.3s;
            box-shadow: 0 8px 20px rgba(0,0,0,0.1);
            margin: 1rem;
            height: 100%;
        }
        .card:hover {
            transform: translateY(-10px);
            box-shadow: 0 15px 30px rgba(0,0,0,0.2);
        }
        h1, h2, h3 {
            color: #1a1a2e;
        }
        .stButton button {
            background: #00b4db;
            color: white;
            border-radius: 30px;
            border: none;
            padding: 0.5rem 1.5rem;
            transition: 0.3s;
        }
        .stButton button:hover {
            background: #0083b0;
            transform: scale(1.02);
        }
        .stFileUploader {
            background: rgba(0,0,0,0.05);
            border-radius: 15px;
            padding: 0.5rem;
        }
        .success-text {
            color: #008080;
            font-weight: bold;
        }
        .error-text {
            color: #cc0000;
            font-weight: bold;
        }
        </style>
        """

# Helper: process image frame
def process_frame(frame):
    eligible, reason, annotated = checker.check(frame)
    return eligible, reason, annotated

# Helper: process video file
def process_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    tfile.close()
    
    cap = cv2.VideoCapture(tfile.name)
    results = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = st.progress(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 15 == 0:
            eligible, reason, _ = checker.check(frame)
            results.append((eligible, reason))
        frame_count += 1
        if total_frames > 0:
            progress.progress(min(frame_count / total_frames, 1.0))
    cap.release()
    os.unlink(tfile.name)
    progress.empty()
    
    if not results:
        return False, "No frames processed"
    for eligible, reason in results:
        if not eligible:
            return False, reason
    return True, "All frames eligible"

# ==================== PAGE DEFINITIONS ====================

def home_page():
    st.markdown(get_css(), unsafe_allow_html=True)
    
    # Header with theme toggle
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("📝 Exam Eligibility Checker")
        st.markdown("*AI-powered proctoring assistant*")
    with col2:
        st.button("🌓 Toggle Theme", on_click=toggle_theme)
    
    # Three cards as navigation buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📸 Upload Photo")
        st.markdown("Submit a single image of your exam room")
        if st.button("Go to Photo Upload", key="go_photo"):
            st.switch_page("streamlit_app.py")  # We'll use query params to simulate page
            # Actually we'll set session state to change page
            st.session_state.page = "photo"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🎥 Upload Video")
        st.markdown("Upload a video recording of your room")
        if st.button("Go to Video Upload", key="go_video"):
            st.session_state.page = "video"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🎥 Live Webcam")
        st.markdown("Real-time face & object detection")
        if st.button("Go to Live Webcam", key="go_webcam"):
            st.session_state.page = "webcam"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("*Ensure only one face is visible, no phones/books/laptops/papers.*")

def photo_page():
    st.markdown(get_css(), unsafe_allow_html=True)
    
    # Header with navigation back and theme toggle
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if st.button("← Back to Home"):
            st.session_state.page = "home"
            st.rerun()
    with col2:
        st.title("📸 Photo Upload")
    with col3:
        st.button("🌓 Toggle Theme", on_click=toggle_theme, key="photo_theme")
    
    st.markdown("Submit a single image of your exam room")
    uploaded_photo = st.file_uploader("Choose an image", type=['jpg','jpeg','png'], label_visibility="collapsed")
    if uploaded_photo is not None:
        image = Image.open(uploaded_photo)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        with st.spinner("Analyzing..."):
            eligible, reason, annotated = process_frame(frame)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption="Analysis Result", use_container_width=True)
        if eligible:
            st.markdown(f'<p class="success-text">✅ ELIGIBLE – {reason}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="error-text">❌ NOT ELIGIBLE – {reason}</p>', unsafe_allow_html=True)

def video_page():
    st.markdown(get_css(), unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if st.button("← Back to Home"):
            st.session_state.page = "home"
            st.rerun()
    with col2:
        st.title("🎥 Video Upload")
    with col3:
        st.button("🌓 Toggle Theme", on_click=toggle_theme, key="video_theme")
    
    st.markdown("Upload a video recording of your room")
    uploaded_video = st.file_uploader("Choose a video", type=['mp4','avi','mov'], label_visibility="collapsed")
    if uploaded_video is not None:
        with st.spinner("Processing video (may take a while)..."):
            eligible, reason = process_video(uploaded_video)
        if eligible:
            st.markdown(f'<p class="success-text">✅ ELIGIBLE – {reason}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="error-text">❌ NOT ELIGIBLE – {reason}</p>', unsafe_allow_html=True)

def webcam_page():
    st.markdown(get_css(), unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if st.button("← Back to Home"):
            st.session_state.page = "home"
            st.rerun()
    with col2:
        st.title("🎥 Live Webcam")
    with col3:
        st.button("🌓 Toggle Theme", on_click=toggle_theme, key="webcam_theme")
    
    st.markdown("Real-time face & object detection")
    captured = st.camera_input("Take a picture", key="webcam_capture")
    if captured is not None:
        image = Image.open(captured)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        with st.spinner("Checking..."):
            eligible, reason, annotated = process_frame(frame)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption="Captured Frame", use_container_width=True)
        if eligible:
            st.markdown(f'<p class="success-text">✅ ELIGIBLE – {reason}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="error-text">❌ NOT ELIGIBLE – {reason}</p>', unsafe_allow_html=True)

# ==================== PAGE ROUTING ====================

if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "photo":
    photo_page()
elif st.session_state.page == "video":
    video_page()
elif st.session_state.page == "webcam":
    webcam_page()