import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Page configuration
st.set_page_config(
    page_title="Proctoring AI - Mobile Detection",
    layout="centered"
)

# Custom CSS for a professional white/material theme
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #ffffff;
        color: #1a1a1a;
    }
    
    /* Center the content */
    .block-container {
        padding-top: 5rem;
        max-width: 800px;
    }

    h1 {
        font-weight: 600;
        color: #1a1a1a;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }

    .developer-credit {
        position: fixed;
        bottom: 20px;
        right: 20px;
        font-size: 0.85rem;
        color: #666;
        background: rgba(255, 255, 255, 0.8);
        padding: 5px 12px;
        border-radius: 20px;
        border: 1px solid #eee;
        z-index: 1000;
        text-decoration: none;
        transition: 0.2s;
    }
    .developer-credit:hover {
        color: #000;
        border-color: #ddd;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        background-color: #1a1a1a;
        color: white !important;
        font-weight: 500;
        border: none;
        transition: all 0.2s ease;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.9rem;
        margin-top: 1.5rem;
    }
    .stButton>button:hover {
        background-color: #333;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transform: translateY(-1px);
    }
    .stButton>button:active {
        transform: translateY(0);
    }

    /* Card/Module Styling */
    .content-box {
        padding: 2.5rem;
        border-radius: 12px;
        border: 1px solid #eaeaea;
        background-color: #fafafa;
        margin-top: 2rem;
    }

    /* Status Indicators */
    .stAlert {
        border-radius: 8px;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# Developer Footer
st.markdown('<a href="https://vishnucax.github.io" target="_blank" class="developer-credit">Developed by Vishnu K</a>', unsafe_allow_html=True)

# Initialize Session State
if 'session_active' not in st.session_state:
    st.session_state.session_active = False
if 'detected' not in st.session_state:
    st.session_state.detected = False

# Load YOLOv8 model
@st.cache_resource
def load_model():
    # Using Small model for better reliability than Nano
    return YOLO('yolov8s.pt')

model = load_model()

def main():
    if not st.session_state.session_active and not st.session_state.detected:
        # Landing Page
        st.title("Automated Proctoring System")
        st.write("Ensuring exam integrity through advanced Computer Vision detection.")
        
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.markdown("""
        ### Instructions
        - Click the start button below to begin monitoring.
        - Ensure your surroundings are clear of unauthorized devices.
        - The session will terminate automatically if a mobile phone is detected in the frame.
        """)
        if st.button("Start Detection Session"):
            st.session_state.session_active = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    elif st.session_state.session_active:
        st.title("Monitoring Active")
        st.warning("Position yourself in front of the camera. The system will process each snapshot for unauthorized devices.")
        
        # Use Streamlit's built-in camera input for cloud compatibility
        enable_camera = st.checkbox("Enable Camera", value=True)
        img_file_buffer = st.camera_input("Take a snapshot for verification", disabled=not enable_camera)
        
        if st.button("End Manual Session"):
            st.session_state.session_active = False
            st.rerun()

        if img_file_buffer is not None:
            # Convert the file buffer to an OpenCV image
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            # YOLO detection (extreme sensitivity at 0.15)
            results = model(cv2_img, stream=True, classes=[67], conf=0.15)
            
            phone_detected = False
            for r in results:
                if len(r.boxes) > 0:
                    phone_detected = True
                    break

            if phone_detected:
                # Play alert sound (system beep implementation for Streamlit)
                st.markdown("""
                    <audio autoplay>
                        <source src="https://www.soundjay.com/buttons/beep-01a.mp3" type="audio/mpeg">
                    </audio>
                """, unsafe_allow_html=True)
                
                st.session_state.session_active = False
                st.session_state.detected = True
                time.sleep(1) # Allow sound to start
                st.rerun()
            else:
                st.success("No unauthorized devices detected in this frame.")

    elif st.session_state.detected:
        # Detected Screen
        st.title("Security Violation")
        st.error("Unauthorized mobile device detected. Session has been terminated.")
        
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.write("A mobile phone or its component was identified in the monitoring area.")
        if st.button("Reset Proctoring System"):
            st.session_state.detected = False
            st.session_state.session_active = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
