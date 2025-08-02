import cv2
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import time
import csv
import os
from datetime import datetime
from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException
import queue
import configparser
from multiprocessing import Process, Queue

# Define DUMMY_CLASSIFICATION_CLASSES globally
DUMMY_CLASSIFICATION_CLASSES = ["no_defect", "contaminate", "crack", "pe_surface", "wrinkle"]

# Configuration Loader
config = configparser.ConfigParser()
config.read('config.ini')

# PLC Configuration (from config.ini)
PLC_IP = config.get('PLC_SETTINGS', 'PLC_IP', fallback="192.168.1.10")
PLC_PORT = config.getint('PLC_SETTINGS', 'PLC_PORT', fallback=502)
PLC_DEFECT_REGISTER = config.getint('PLC_SETTINGS', 'PLC_DEFECT_REGISTER', fallback=1)
PLC_ACK_REGISTER = config.getint('PLC_SETTINGS', 'PLC_ACK_REGISTER', fallback=2)

class Camera:
    """
    A class to manage camera (or video file) access using OpenCV.
    """
    def __init__(self, source=0, width=640, height=480):
        self.source = source
        self.width = width
        self.height = height
        self.cap = None

    def open(self, error_queue=None):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            error_message = f"Error: Could not open video source {self.source}. Please check camera connection or video path."
            print(error_message)
            if error_queue:
                try: error_queue.put_nowait(error_message)
                except queue.Full: pass
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        print(f"Camera/Video source {self.source} opened successfully.")
        print(f"Resolution set to {self.width}x{self.height}")
        return True

    def read_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            return ret, frame
        return False, None

    def release(self):
        if self.cap:
            self.cap.release()
            print(f"Camera/Video source {self.source} released.")

def create_defect_classifier_model(input_shape, num_classes):
    """
    Creates a Convolutional Neural Network (CNN) model for defect classification.
    This is a placeholder model and should be customized and trained with actual data.
    For better performance, consider more complex architectures or pre-trained models with transfer learning.
    """
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_object_detection_model(input_shape, num_classes, num_boxes_per_grid=1):
    """
    Creates a placeholder Convolutional Neural Network (CNN) model for object detection.
    This is a simplified YOLO-like structure, suitable for demonstrating localization.
    The output of the model will be a grid tensor, which can then be parsed to extract
    bounding boxes and class probabilities.

    Input_shape: Shape of the input image (height, width, channels).
    num_classes: Number of defect classes.
    num_boxes_per_grid: Number of bounding box predictions per grid cell.

    The output tensor shape will be (batch_size, grid_H, grid_W, num_boxes_per_grid * (5 + num_classes))
    For each bounding box: (x, y, w, h, confidence, class_probabilities...)
    """
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(num_boxes_per_grid * (5 + num_classes), (1, 1), activation='sigmoid')
    ])
    return model

# New configuration for model saving
MODEL_SAVE_DIR = config.get('MODEL_SETTINGS', 'MODEL_SAVE_DIR', fallback="models")
CLASSIFICATION_MODEL_NAME = config.get('MODEL_SETTINGS', 'CLASSIFICATION_MODEL_NAME', fallback="defect_classifier_model.h5")
OBJECT_DETECTION_MODEL_NAME = config.get('MODEL_SETTINGS', 'OBJECT_DETECTION_MODEL_NAME', fallback="object_detection_model.h5")

def train_model(model, data_dir, input_shape, num_classes, epochs=10, model_type="classification"):
    """
    Trains the given model with data loaded from the specified directory.
    Assumes `data_dir` contains subdirectories, each representing a class.
    Saves the trained model to a file.
    """
    print(f"\n--- Training {model_type.capitalize()} Model ---")

    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}. Please create it and add images.")
        return None

    try:
        train_ds = keras.utils.image_dataset_from_directory(
            data_dir,
            labels='inferred',
            label_mode='int',
            image_size=(input_shape[0], input_shape[1]),
            interpolation='nearest',
            batch_size=32,
            shuffle=True
        )

        data_augmentation = keras.models.Sequential([
            keras.layers.RandomFlip("horizontal_and_vertical"),
            keras.layers.RandomRotation(0.2),
            keras.layers.RandomZoom(0.2),
            keras.layers.RandomContrast(0.2),
        ])

        augmented_train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
        normalization_layer = keras.layers.Rescaling(1./255)
        normalized_train_ds = augmented_train_ds.map(lambda x, y: (normalization_layer(x), y))

        AUTOTUNE = tf.data.AUTOTUNE
        normalized_train_ds = normalized_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

        model.fit(normalized_train_ds, epochs=epochs)

        if not os.path.exists(MODEL_SAVE_DIR):
            os.makedirs(MODEL_SAVE_DIR)
        
        if model_type == "classification":
            save_path = os.path.join(MODEL_SAVE_DIR, CLASSIFICATION_MODEL_NAME)
        elif model_type == "object_detection":
            save_path = os.path.join(MODEL_SAVE_DIR, OBJECT_DETECTION_MODEL_NAME)
        else:
            print("Warning: Unknown model type. Model not saved.")
            return model

        model.save(save_path)
        print(f"--- {model_type.capitalize()} Model Training Complete and Saved to {save_path} ---")
        return model

    except Exception as e:
        print(f"Error during model training or data loading: {e}")
        print("Please ensure your data directory structure is correct (e.g., `data_dir/class_1/...`, `data_dir/class_2/...`)")
        return None

DEFECT_LOG_FILE = "defect_log.csv"
DEFECT_IMAGE_DIR = "defective_images"

def initialize_defect_log():
    """
    Initializes the CSV log file with headers if it doesn't already exist.
    Also creates the directory for defective images.
    """
    if not os.path.exists(DEFECT_LOG_FILE):
        with open(DEFECT_LOG_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Defect_Class", "Confidence", "Image_Width", "Image_Height", "X_Coord", "Y_Coord", "Bounding_Box", "Image_Path"])
        print(f"Created new defect log file: {DEFECT_LOG_FILE}")
    else:
        print(f"Defect log file already exists: {DEFECT_LOG_FILE}")

    if not os.path.exists(DEFECT_IMAGE_DIR):
        os.makedirs(DEFECT_IMAGE_DIR)
        print(f"Created directory for defective images: {DEFECT_IMAGE_DIR}")
    else:
        print(f"Defect image directory already exists: {DEFECT_IMAGE_DIR}")

def log_defect(defect_class, confidence, image_shape, defect_location=None, bounding_box=None, image_path=None, log_queue=None):
    """
    Logs a detected defect to the CSV file.
    image_shape should be (height, width, channels).
    defect_location should be (x, y) coordinates of the defect (e.g., center).
    bounding_box should be (x_min, y_min, x_max, y_max) for the defect.
    image_path is the path to the saved image of the defect.
    log_queue (optional): A queue to send log data to (e.g., for Streamlit).
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    height, width, _ = image_shape
    x_coord, y_coord = "", ""
    if defect_location:
        x_coord, y_coord = defect_location

    bbox_str = ""
    if bounding_box:
        bbox_str = f"{bounding_box[0]},{bounding_box[1]},{bounding_box[2]},{bounding_box[3]}"

    log_entry = {
        "Timestamp": timestamp,
        "Defect_Class": defect_class,
        "Confidence": f"{confidence:.2f}",
        "Image_Width": width,
        "Image_Height": height,
        "X_Coord": x_coord,
        "Y_Coord": y_coord,
        "Bounding_Box": bbox_str,
        "Image_Path": image_path
    }

    with open(DEFECT_LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(list(log_entry.values()))
    print(f"Logged defect: Class {defect_class}, Confidence {confidence:.2f} at ({x_coord}, {y_coord}) BBox: {bbox_str} Image: {image_path}")

    if log_queue:
        try:
            log_queue.put_nowait(log_entry)
        except queue.Full:
            print("Warning: Log queue is full, skipping log entry for Streamlit.")

def log_writer_process(data_queue, log_file, image_dir, ui_log_queue=None):
    """
    Worker process to write defect logs to CSV and save images asynchronously.
    """
    print("Log writer process started.")
    while True:
        try:
            log_data = data_queue.get(timeout=1)
            if log_data == "STOP":
                print("Log writer process stopping.")
                break

            defect_class = log_data["defect_class"]
            confidence = log_data["confidence"]
            image_shape = log_data["image_shape"]
            defect_location = log_data.get("defect_location")
            bounding_box = log_data.get("bounding_box")
            image_path = log_data.get("image_path")
            frame_to_save = log_data.get("frame_to_save")

            if image_path and frame_to_save is not None:
                cv2.imwrite(image_path, frame_to_save)
                print(f"Saved defective image to: {image_path}")

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            height, width, _ = image_shape
            x_coord, y_coord = "", ""
            if defect_location:
                x_coord, y_coord = defect_location

            bbox_str = ""
            if bounding_box:
                bbox_str = f"{bounding_box[0]},{bounding_box[1]},{bounding_box[2]},{bounding_box[3]}"

            log_entry_csv = {
                "Timestamp": timestamp,
                "Defect_Class": defect_class,
                "Confidence": f"{confidence:.2f}",
                "Image_Width": width,
                "Image_Height": height,
                "X_Coord": x_coord,
                "Y_Coord": y_coord,
                "Bounding_Box": bbox_str,
                "Image_Path": image_path
            }
            with open(log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(list(log_entry_csv.values()))
            print(f"Logged defect (async): Class {defect_class}, Confidence {confidence:.2f} at ({x_coord}, {y_coord}) BBox: {bbox_str} Image: {image_path}")

            if ui_log_queue:
                try:
                    ui_log_queue.put_nowait(log_entry_csv)
                except queue.Full:
                    print("Warning: UI Log queue is full, skipping log entry.")

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in log_writer_process: {e}")

def plc_commander_process(command_queue, plc_ip, plc_port, ui_plc_queue=None, ui_error_queue=None):
    """
    Worker process to send PLC commands asynchronously.
    """
    print("PLC commander process started.")
    while True:
        try:
            command_data = command_queue.get(timeout=1)
            if command_data == "STOP":
                print("PLC commander process stopping.")
                break

            register_address = command_data["register_address"]
            value = command_data["value"]
            command_type = command_data["command_type"]

            send_plc_command(register_address, value, plc_ip, plc_port, command_type, ui_plc_queue, ui_error_queue)

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in plc_commander_process: {e}")

def send_plc_command(register_address, value, plc_ip=PLC_IP, plc_port=PLC_PORT, command_type="coil", plc_queue=None, error_queue=None):
    """
    Sends a command to the PLC via Modbus TCP with improved error handling.
    - register_address: The Modbus register address to write to.
    - value: The value to write to the register.
    - plc_ip: The IP address of the PLC.
    - plc_port: The Modbus TCP port of the PLC.
    - command_type: "coil" for write_coil, "holding_register" for write_register, etc.
    - plc_queue (optional): A queue to send PLC command status to (e.g., for Streamlit).
    - error_queue (optional): A queue to send error messages to (e.g., for Streamlit).
    """
    client = None
    try:
        client = ModbusTcpClient(plc_ip, port=plc_port, timeout=1)
        if client.connect():
            status_message = f"Connected to PLC at {plc_ip}:{plc_port}"
            print(status_message)
            if plc_queue:
                try:
                    plc_queue.put_nowait(status_message)
                except queue.Full:
                    pass

            result = None

            if command_type == "coil":
                result = client.write_coil(register_address, value)
            elif command_type == "holding_register":
                result = client.write_register(register_address, value)
            else:
                status_message = f"Error: Unsupported Modbus command type: {command_type}"
                print(status_message)
                if plc_queue:
                    try: plc_queue.put_nowait(status_message)
                    except queue.Full: pass
                if error_queue:
                    try: error_queue.put_nowait(status_message)
                    except queue.Full: pass
                return

            if result is not None and result.isError():
                status_message = f"Modbus Error writing to register {register_address}: {result}"
                print(status_message)
                if plc_queue:
                    try: plc_queue.put_nowait(status_message)
                    except queue.Full: pass
                if error_queue:
                    try: error_queue.put_nowait(status_message)
                    except queue.Full: pass
            elif result is not None:
                status_message = f"Sent command to PLC: Set register {register_address} to {value} (Type: {command_type})"
                print(status_message)
                if plc_queue:
                    try: plc_queue.put_nowait(status_message)
                    except queue.Full: pass

        else:
            status_message = f"Failed to connect to PLC at {plc_ip}:{plc_port}"
            print(status_message)
            if plc_queue:
                try: plc_queue.put_nowait(status_message)
                except queue.Full: pass
            if error_queue:
                try: error_queue.put_nowait(status_message)
                except queue.Full: pass
    except ModbusException as e:
        status_message = f"Modbus Communication Error: {e}"
        print(status_message)
        if plc_queue:
            try: plc_queue.put_nowait(status_message)
            except queue.Full: pass
        if error_queue:
            try: error_queue.put_nowait(status_message)
            except queue.Full: pass
    except Exception as e:
        status_message = f"An unexpected error occurred during PLC communication: {e}"
        print(status_message)
        if plc_queue:
            try: plc_queue.put_nowait(status_message)
            except queue.Full: pass
        if error_queue:
            try: error_queue.put_nowait(status_message)
            except queue.Full: pass
    finally:
        if client:
            client.close()

def process_video_feed(video_source=0, model=None, log_queue=None, plc_command_queue=None, error_queue=None, log_data_queue=None, plc_command_data_queue=None, object_detection_model=None):
    """
    Captures video from a source (camera or file), processes frames for defect classification,
    and sends results to provided queues for display in a UI.
    No direct display (cv2.imshow) is done here.
    """
    camera = Camera(source=video_source)
    if not camera.open(error_queue=error_queue):
        if error_queue:
            try: error_queue.put_nowait(f"Error: Could not open video source {video_source}. Aborting video processing.")
            except queue.Full: pass
        return

    print("\n--- Starting Real-time Video Feed Processing ---")
    while True:
        ret, frame = camera.read_frame()

        if not ret:
            error_message = "End of video stream or error reading frame."
            print(error_message)
            if error_queue:
                try: error_queue.put_nowait(error_message)
                except queue.Full: pass
            break

        if frame is None:
            error_message = "Received empty frame, skipping processing."
            print(error_message)
            if error_queue:
                try: error_queue.put_nowait(error_message)
                except queue.Full: pass
            continue

        processed_frame_for_model = frame.copy()

        if model:
            processed_frame_for_classification = cv2.resize(processed_frame_for_model, (64, 64))
            processed_frame_for_classification = cv2.cvtColor(processed_frame_for_classification, cv2.COLOR_BGR2RGB) / 255.0
            processed_frame_for_classification = np.expand_dims(processed_frame_for_classification, axis=0)

            predictions_classification = model.predict(processed_frame_for_classification)
            predicted_class = np.argmax(predictions_classification[0])
            confidence = np.max(predictions_classification[0])

            # Debugging print statements
            print(f"[DEBUG] Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")

            bounding_box = None
            if object_detection_model and predicted_class > 0:
                print("[DEBUG] Object detection model is active.")
                processed_frame_for_detection = cv2.resize(processed_frame_for_model, object_detection_model.input_shape[1:3])
                processed_frame_for_detection = cv2.cvtColor(processed_frame_for_detection, cv2.COLOR_BGR2RGB) / 255.0
                processed_frame_for_detection = np.expand_dims(processed_frame_for_detection, axis=0)

                predictions_detection = object_detection_model.predict(processed_frame_for_detection)
                
                _, grid_h, grid_w, _ = object_detection_model.output_shape
                
                if predictions_detection.shape[1] > 0 and predictions_detection.shape[2] > 0:
                    box_prediction = predictions_detection[0, 0, 0, :]
                    center_x_norm, center_y_norm, width_norm, height_norm, confidence_obj_det = box_prediction[0:5]
                
                    frame_height, frame_width, _ = frame.shape
                    center_x_px = int(center_x_norm * frame_width)
                    center_y_px = int(center_y_norm * frame_height)
                    width_px = int(width_norm * frame_width)
                    height_px = int(height_norm * frame_height)

                    x_min = int(center_x_px - (width_px / 2))
                    y_min = int(center_y_px - (height_px / 2))
                    x_max = int(center_x_px + (width_px / 2))
                    y_max = int(center_y_px + (height_px / 2))

                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(frame_width - 1, x_max)
                    y_max = min(frame_height - 1, y_max)

                    bounding_box = (x_min, y_min, x_max, y_max)
                    print(f"[DEBUG] Bounding Box Generated: {bounding_box}")
                else:
                    print("[DEBUG] Warning: Object detection model output has no spatial dimensions or is empty.")

            saved_image_path = None
            if predicted_class > 0 and bounding_box:
                print("[DEBUG] Condition for logging met: predicted_class > 0 and bounding_box is not None.")
                cv2.rectangle(frame, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (0, 0, 255), 2)

                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                image_filename = f"defect_{predicted_class}_{timestamp_str}.jpg"
                saved_image_path = os.path.join(DEFECT_IMAGE_DIR, image_filename)
                
                if log_queue:
                    log_entry_for_ui = {
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Defect_Class": DUMMY_CLASSIFICATION_CLASSES[predicted_class],
                        "Confidence": f"{confidence:.2f}",
                        "Image_Width": frame_width,
                        "Image_Height": frame_height,
                        "X_Coord": (bounding_box[0] + bounding_box[2]) // 2 if bounding_box else "",
                        "Y_Coord": (bounding_box[1] + bounding_box[3]) // 2 if bounding_box else "",
                        "Bounding_Box": f"{bounding_box[0]},{bounding_box[1]},{bounding_box[2]},{bounding_box[3]}" if bounding_box else "",
                        "Image_Path": saved_image_path
                    }
                    try:
                        log_queue.put_nowait(log_entry_for_ui)
                        print("[DEBUG] Log entry put into UI log queue.")
                    except queue.Full:
                        print("Warning: UI Log queue is full, skipping log entry.")

                if log_data_queue:
                    log_data_for_worker = {
                        "defect_class": DUMMY_CLASSIFICATION_CLASSES[predicted_class],
                        "confidence": confidence,
                        "image_shape": frame.shape,
                        "defect_location": ((bounding_box[0] + bounding_box[2]) // 2, (bounding_box[1] + bounding_box[3]) // 2) if bounding_box else None,
                        "bounding_box": bounding_box,
                        "image_path": saved_image_path,
                        "frame_to_save": frame.copy()
                    }
                    try:
                        log_data_queue.put_nowait(log_data_for_worker)
                        print("[DEBUG] Log data put into internal logging queue.")
                    except queue.Full:
                        print("Warning: Internal logging queue is full, skipping detailed log entry.")

            if predicted_class > 0:
                print("[DEBUG] Predicted class > 0, checking PLC command.")
                if plc_command_data_queue:
                    plc_command_for_worker = {
                        "register_address": PLC_DEFECT_REGISTER,
                        "value": True,
                        "command_type": "coil"
                    }
                    try:
                        plc_command_data_queue.put_nowait(plc_command_for_worker)
                        print("[DEBUG] PLC command put into internal PLC command queue.")
                    except queue.Full:
                        print("Warning: Internal PLC command queue is full, skipping PLC command.")

            text = f"Defect: {predicted_class} (Conf: {confidence:.2f})"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield rgb_frame
    
    camera.release()
    print("--- Video Feed Processing Stopped ---")

if __name__ == "__main__":
    input_shape = (64, 64, 3)
    num_defect_classes = 5

    DUMMY_TRAIN_DATA_DIR = "training_data"
    # Removed DUMMY_CLASSIFICATION_CLASSES from here, now global
    
    if not os.path.exists(DUMMY_TRAIN_DATA_DIR):
        os.makedirs(DUMMY_TRAIN_DATA_DIR)
        print(f"Created dummy training data directory: {DUMMY_TRAIN_DATA_DIR}")
        for class_name in DUMMY_CLASSIFICATION_CLASSES:
            class_dir = os.path.join(DUMMY_TRAIN_DATA_DIR, class_name)
            os.makedirs(class_dir, exist_ok=True)
            for i in range(5):
                dummy_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
                cv2.imwrite(os.path.join(class_dir, f"dummy_image_{class_name}_{i}.png"), dummy_image)
        print("Populated dummy training data with placeholder images.")
    else:
        print(f"Dummy training data directory already exists: {DUMMY_TRAIN_DATA_DIR}")

    defect_model = create_defect_classifier_model(input_shape, num_defect_classes)
    
    print("\nAttempting to train classification model with dummy data...")
    defect_model = train_model(defect_model, DUMMY_TRAIN_DATA_DIR, input_shape, num_defect_classes, epochs=1, model_type="classification")
    
    object_detection_model = create_object_detection_model(input_shape, num_defect_classes)
    print("\nAttempting to train object detection model with dummy data...")
    object_detection_model = train_model(object_detection_model, DUMMY_TRAIN_DATA_DIR, input_shape, num_defect_classes, epochs=1, model_type="object_detection")

    initialize_defect_log()

    trained_classification_model_path = os.path.join(MODEL_SAVE_DIR, CLASSIFICATION_MODEL_NAME)
    if os.path.exists(trained_classification_model_path):
        print(f"Loading trained classification model from {trained_classification_model_path}")
        defect_model = keras.models.load_model(trained_classification_model_path)
    else:
        print("No trained classification model found. Using newly created model (untrained).")

    trained_object_detection_model_path = os.path.join(MODEL_SAVE_DIR, OBJECT_DETECTION_MODEL_NAME)
    if os.path.exists(trained_object_detection_model_path):
        print(f"Loading trained object detection model from {trained_object_detection_model_path}")
        object_detection_model = keras.models.load_model(trained_object_detection_model_path)
    else:
        print("No trained object detection model found. Using newly created model (untrained).")

    internal_log_data_queue = Queue()
    internal_plc_command_data_queue = Queue()
    internal_error_queue = Queue()

    log_writer_proc = Process(
        target=log_writer_process,
        args=(internal_log_data_queue, DEFECT_LOG_FILE, DEFECT_IMAGE_DIR, None)
    )
    log_writer_proc.start()

    plc_commander_proc = Process(
        target=plc_commander_process,
        args=(internal_plc_command_data_queue, PLC_IP, PLC_PORT, None, internal_error_queue)
    )
    plc_commander_proc.start()

    video_source = 0
    for frame in process_video_feed(video_source, defect_model, log_queue=None, plc_command_queue=None, error_queue=internal_error_queue, log_data_queue=internal_log_data_queue, plc_command_data_queue=internal_plc_command_data_queue, object_detection_model=object_detection_model):
        if not internal_error_queue.empty():
            err_msg = internal_error_queue.get()
            print(f"Internal Process Error: {err_msg}")

        cv2.imshow("Test Feed (main.py)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    internal_log_data_queue.put("STOP")
    internal_plc_command_data_queue.put("STOP")
    log_writer_proc.join()
    plc_commander_proc.join()

    cv2.destroyAllWindows() 