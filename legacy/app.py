import streamlit as st
import cv2 # noqa: F401
import numpy as np
# Removed multiprocessing.Process, Queue as they are now encapsulated in VideoProcessor
# from multiprocessing import Process, Queue
import time
import pandas as pd
import os
from datetime import datetime
# Removed tensorflow and keras imports as they are in VideoProcessor
# import tensorflow as tf
# from tensorflow import keras
import shutil # Re-added shutil import for os.remove and shutil.rmtree in cleanup

# Import functions/classes and configurations from main.py
from src.main import (
    Camera,
    create_defect_classifier_model, # Keep if still needed for dummy model or other logic
    create_object_detection_model, # Keep if still needed for dummy model or other logic
    process_video_feed, # Keep as it's used in run_process_video_feed
    initialize_defect_log,
    PLC_DEFECT_REGISTER,
    PLC_IP,
    PLC_PORT,
    send_plc_command,
    log_writer_process,
    plc_commander_process,
    DEFECT_LOG_FILE,
    DEFECT_IMAGE_DIR,
    MODEL_SAVE_DIR,
    CLASSIFICATION_MODEL_NAME,
    OBJECT_DETECTION_MODEL_NAME
)

# Import the new VideoProcessor
from src.video_processing import VideoProcessor

# Queues for inter-process communication - moved to VideoProcessor
# frame_queue = Queue(maxsize=5)
# defect_log_queue = Queue(maxsize=100)
# plc_status_queue = Queue(maxsize=100)
# error_queue = Queue(maxsize=100)

# Internal queues for worker processes - moved to VideoProcessor
# internal_log_data_queue = Queue()
# internal_plc_command_data_queue = Queue()


# Session state for managing the process
# if 'process' not in st.session_state:
#     st.session_state.process = None
# if 'running' not in st.session_state:
#     st.session_state.running = False

# Initialize defect_df in session state
if 'defect_df' not in st.session_state:
    st.session_state.defect_df = pd.DataFrame(columns=["Timestamp", "Defect_Class", "Confidence", "Image_Width", "Image_Height", "X_Coord", "Y_Coord", "Bounding_Box", "Image_Path"])

# Removed old start_processing and run_process_video_feed functions
# def start_processing(...):
# def run_process_video_feed(...):

# Function to load defect log (for display on stop/initial load)
def load_defect_log(file_path=DEFECT_LOG_FILE):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df
    return pd.DataFrame()

# Streamlit UI
st.title("Image Processing Defect Detection")

# Initialize VideoProcessor
if 'video_processor' not in st.session_state:
    st.session_state.video_processor = VideoProcessor()
video_processor = st.session_state.video_processor

# Sidebar for configuration
st.sidebar.header("Configuration")
video_source_option = st.sidebar.radio(
    "Select Video Source:",
    ("Video File", "Webcam", "IP Camera")
)

video_source_input = None
uploaded_file = None

if video_source_option == "Webcam":
    video_source_input = st.sidebar.number_input("Webcam Index (e.g., 0, 1)", min_value=0, value=0, step=1)
elif video_source_option == "Video File":
    uploaded_file = st.sidebar.file_uploader("Upload Video File", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        temp_dir = "temp_videos"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        video_source_input = temp_file_path
        st.sidebar.info(f"Video file uploaded and saved to: {uploaded_file.name}")
    else:
        st.sidebar.info("Please upload a video file.")
elif video_source_option == "IP Camera":
    video_source_input = st.sidebar.text_input("IP Camera URL (e.g., rtsp://user:pass@ip:port/stream)")

if st.sidebar.button("Start Processing", disabled=video_processor.running):
    if video_processor.process is None or not video_processor.process.is_alive():
        if video_source_input is None or (isinstance(video_source_input, str) and not video_source_input.strip() and video_source_option != "Video File") or (video_source_option == "Video File" and uploaded_file is None):
            st.sidebar.warning("Please provide a valid video source to start.")
        else:
            initialize_defect_log()

            # Load models using VideoProcessor
            classification_model, object_detection_model = video_processor.load_models()
            
            video_processor.start_processing(video_source=video_source_input, classification_model=classification_model, object_detection_model=object_detection_model)
            st.success("Video processing started.")

if st.sidebar.button("Stop Processing", disabled=not video_processor.running):
    # Use video_processor.stop_processing
    video_processor.stop_processing(video_source_option, uploaded_file)
    st.sidebar.info("Video processing stopped.")


# Main content area
st.subheader("Real-time Camera Feed")
video_placeholder = st.empty()

st.subheader("Defect Log")
defect_log_placeholder = st.empty()

st.subheader("PLC Status")
plc_status_placeholder = st.empty()
plc_messages = []

st.subheader("Application Errors")
error_placeholder = st.empty()
error_messages = []

# Update loop for Streamlit UI
while video_processor.running:
    if not video_processor.get_frame_queue().empty():
        frame = video_processor.get_frame_queue().get()
        video_placeholder.image(frame, channels="RGB", use_container_width=True) # Changed from use_column_width

    # Update defect log
    new_log_entries = []
    while not video_processor.get_defect_log_queue().empty():
        new_log_entries.append(video_processor.get_defect_log_queue().get())
    if new_log_entries:
        temp_df = pd.DataFrame(new_log_entries)
        if 'Timestamp' in temp_df.columns:
            temp_df['Timestamp'] = pd.to_datetime(temp_df['Timestamp'])
        
        # Using st.session_state to manage defect_df
        st.session_state.defect_df = pd.concat([st.session_state.defect_df, temp_df], ignore_index=True)
        defect_log_placeholder.dataframe(st.session_state.defect_df)
    
    # Update PLC status
    new_plc_messages = []
    while not video_processor.get_plc_status_queue().empty():
        new_plc_messages.append(video_processor.get_plc_status_queue().get())
    if new_plc_messages:
        plc_messages.extend(new_plc_messages)
        plc_status_placeholder.text("\n".join(plc_messages[-5:])) # Display last 5 messages

    # Update error messages
    new_error_messages = []
    while not video_processor.get_error_queue().empty():
        new_error_messages.append(video_processor.get_error_queue().get())
    if new_error_messages:
        error_messages.extend(new_error_messages)
        error_placeholder.error("\n".join(error_messages[-5:])) # Display last 5 errors
    
    time.sleep(0.01) # Small delay to prevent busy-waiting and allow UI to update

# Display data when stopped or initially (ensuring content is visible even when not running)
if not video_processor.running:
    # If there are frames left in the queue after stopping, display the last one
    if not video_processor.get_frame_queue().empty():
        frame = video_processor.get_frame_queue().get()
        video_placeholder.image(frame, channels="RGB", use_container_width=True)

    # Display final defect log
    if not st.session_state.defect_df.empty:
        defect_log_placeholder.dataframe(st.session_state.defect_df)
    else:
        defect_log_placeholder.info("No defect entries yet.")

    # Display final PLC status
    if plc_messages:
        plc_status_placeholder.text("\n".join(plc_messages[-5:]))
    else:
        plc_status_placeholder.info("No PLC messages yet.")

    # Display final application errors
    if error_messages:
        error_placeholder.error("\n".join(error_messages[-5:]))
    else:
        error_placeholder.info("No application errors yet.")

# Ensure temporary video directory is cleaned up on app exit
@st.cache_resource(ttl=None)
def setup_cleanup():
    def cleanup_temp_videos():
        if os.path.exists("temp_videos"): # Added check for directory existence
            shutil.rmtree("temp_videos")
            print("Cleaned up temporary video directory on app exit.")
    
    import atexit
    atexit.register(cleanup_temp_videos)
    return True

if setup_cleanup():
    pass 