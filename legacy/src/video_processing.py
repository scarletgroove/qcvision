import os
import time
import pandas as pd
from multiprocessing import Process, Queue
import tensorflow as tf
import tensorflow.keras as keras
import shutil

from src.main import (
    create_defect_classifier_model,
    create_object_detection_model,
    process_video_feed,
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

class VideoProcessor:
    def __init__(self):
        self.frame_queue = Queue(maxsize=5)
        self.defect_log_queue = Queue(maxsize=100)
        self.plc_status_queue = Queue(maxsize=100)
        self.error_queue = Queue(maxsize=100)

        self.internal_log_data_queue = Queue()
        self.internal_plc_command_data_queue = Queue()

        self.process = None
        self.running = False

    def start_processing(self, video_source, classification_model, object_detection_model):
        self.process = Process(
            target=self._run_process_video_feed,
            args=(
                video_source,
                classification_model,
                self.defect_log_queue,
                self.plc_status_queue,
                self.error_queue,
                self.internal_log_data_queue,
                self.internal_plc_command_data_queue,
                object_detection_model
            )
        )
        self.process.start()
        self.running = True

    def stop_processing(self, video_source_option, uploaded_file):
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join()
            self.running = False

        if video_source_option == "Video File" and uploaded_file is not None:
            temp_file_path = os.path.join("temp_videos", uploaded_file.name)
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        if os.path.exists("temp_videos"):
            shutil.rmtree("temp_videos")

    def _run_process_video_feed(self, video_source, model, log_q, plc_q, error_q, log_data_q, plc_command_data_q, object_detection_model):
        log_writer_proc = Process(
            target=log_writer_process,
            args=(log_data_q, DEFECT_LOG_FILE, DEFECT_IMAGE_DIR, log_q)
        )
        log_writer_proc.start()

        plc_commander_proc = Process(
            target=plc_commander_process,
            args=(plc_command_data_q, PLC_IP, PLC_PORT, plc_q, error_q)
        )
        plc_commander_proc.start()

        for frame in process_video_feed(video_source, model, log_q, plc_q, error_q, log_data_q, plc_command_data_q, object_detection_model):
            if not self.frame_queue.full(): # Use self.frame_queue
                self.frame_queue.put(frame) # Use self.frame_queue
            else:
                time.sleep(0.005)
        
        log_data_q.put("STOP")
        plc_command_data_q.put("STOP")
        log_writer_proc.join()
        plc_commander_proc.join()

        print("Processing process finished.")

    def get_frame_queue(self):
        return self.frame_queue

    def get_defect_log_queue(self):
        return self.defect_log_queue

    def get_plc_status_queue(self):
        return self.plc_status_queue

    def get_error_queue(self):
        return self.error_queue

    def load_models(self):
        classification_model_path = os.path.join(MODEL_SAVE_DIR, CLASSIFICATION_MODEL_NAME)
        if os.path.exists(classification_model_path):
            classification_model = keras.models.load_model(classification_model_path)
        else:
            input_shape = (64, 64, 3)
            num_defect_classes = 5
            classification_model = create_defect_classifier_model(input_shape, num_defect_classes)

        object_detection_model_path = os.path.join(MODEL_SAVE_DIR, OBJECT_DETECTION_MODEL_NAME)
        if os.path.exists(object_detection_model_path):
            object_detection_model = keras.models.load_model(object_detection_model_path)
        else:
            input_shape = (64, 64, 3)
            num_defect_classes = 5
            object_detection_model = create_object_detection_model(input_shape, num_defect_classes)
        return classification_model, object_detection_model 