"""
Code Examples: Real-Time Defect Detection
Copy and adapt these examples for your use case
"""

# ==============================================================================
# EXAMPLE 1: Minimal Setup - Works with defaults
# ==============================================================================

from src.realtime_defect_detector import run_realtime_detection

# Run detection on default webcam
run_realtime_detection(video_source=0)

# That's it! Detection starts immediately.
# Logs saved to: defect_log.csv
# Images saved to: defective_images/


# ==============================================================================
# EXAMPLE 2: Custom Configuration
# ==============================================================================

from src.realtime_defect_detector import DefectDetector
import cv2

# Create detector with custom settings
detector = DefectDetector(
    log_file="production_log.csv",
    image_dir="defects_found"
)

# Adjust sensitivity for your paper type
detector.CONFIDENCE_THRESHOLD = 0.65    # Stricter
detector.DARK_THRESHOLD = 45             # More sensitive to dark particles
detector.BRIGHT_THRESHOLD = 190          # More sensitive to dust
detector.MIN_DEFECT_AREA = 15            # Detect smaller defects

# Open video source
cap = cv2.VideoCapture(0)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    marked_frame, defect_info = detector.process_frame(frame)
    
    if defect_info:
        print(f"Defect found: {defect_info['type']} (Confidence: {defect_info['confidence']:.2f})")
    
    # Display
    cv2.imshow("Detection", marked_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_count += 1

cap.release()
cv2.destroyAllWindows()

print(f"Processed {frame_count} frames")


# ==============================================================================
# EXAMPLE 3: Using with IP Camera (for industrial setup)
# ==============================================================================

from src.realtime_defect_detector import run_realtime_detection

# Connect to IP camera on production line
camera_url = "rtsp://admin:password@192.168.1.100:554/stream"

run_realtime_detection(
    video_source=camera_url,
    detection_threshold=0.6
)

# Log saves to same location
# Can run 24/7 for continuous inspection


# ==============================================================================
# EXAMPLE 4: Multiple Video Inputs (if you have multiple inspection points)
# ==============================================================================

from src.realtime_defect_detector import DefectDetector
import cv2
import threading
import datetime

def process_camera(camera_id, video_source, log_suffix):
    """Process single camera in thread"""
    detector = DefectDetector(
        log_file=f"defect_log_{log_suffix}.csv",
        image_dir=f"defects_{log_suffix}"
    )
    
    cap = cv2.VideoCapture(video_source)
    detector_active = True
    
    while detector_active:
        ret, frame = cap.read()
        if not ret:
            break
        
        marked_frame, defect_info = detector.process_frame(frame)
        
        if defect_info:
            print(f"[Camera {camera_id}] {defect_info['type']}: {defect_info['confidence']:.2f}")
        
        cv2.imshow(f"Camera {camera_id}", marked_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            detector_active = False
    
    cap.release()

# Start inspection on multiple cameras
camera_configs = [
    (1, 0, "line1_intake"),
    (2, 1, "line1_midpoint"),
    (3, 2, "line1_output"),
]

threads = []
for camera_id, video_source, suffix in camera_configs:
    thread = threading.Thread(
        target=process_camera,
        args=(camera_id, video_source, suffix),
        daemon=True
    )
    thread.start()
    threads.append(thread)

# Keep running
for thread in threads:
    thread.join()


# ==============================================================================
# EXAMPLE 5: Filtering based on Paper Type
# ==============================================================================

from src.realtime_defect_detector import DefectDetector

# Configuration presets for different paper types
PRESETS = {
    "white_paper": {
        "CONFIDENCE_THRESHOLD": 0.60,
        "DARK_THRESHOLD": 40,
        "BRIGHT_THRESHOLD": 210,
        "MIN_DEFECT_AREA": 20
    },
    "kraft_paper": {
        "CONFIDENCE_THRESHOLD": 0.55,
        "DARK_THRESHOLD": 50,
        "BRIGHT_THRESHOLD": 190,
        "MIN_DEFECT_AREA": 25
    },
    "tissue_paper": {
        "CONFIDENCE_THRESHOLD": 0.50,
        "DARK_THRESHOLD": 60,
        "BRIGHT_THRESHOLD": 180,
        "MIN_DEFECT_AREA": 15
    },
    "premium_quality": {
        "CONFIDENCE_THRESHOLD": 0.70,
        "DARK_THRESHOLD": 60,
        "BRIGHT_THRESHOLD": 200,
        "MIN_DEFECT_AREA": 30
    }
}

def create_detector_for_paper_type(paper_type):
    """Create detector with preset for specific paper type"""
    detector = DefectDetector()
    
    if paper_type in PRESETS:
        settings = PRESETS[paper_type]
        for key, value in settings.items():
            setattr(detector, key, value)
        print(f"Loaded preset: {paper_type}")
    else:
        print(f"Unknown paper type: {paper_type}. Using defaults.")
    
    return detector

# Usage
detector = create_detector_for_paper_type("white_paper")


# ==============================================================================
# EXAMPLE 6: Real-time Analysis & Alerts
# ==============================================================================

from src.realtime_defect_detector import DefectDetector
import cv2
import collections

def run_with_analytics():
    """Run detection with real-time analytics"""
    detector = DefectDetector()
    cap = cv2.VideoCapture(0)
    
    # Track statistics
    defect_counts = collections.defaultdict(int)
    frame_count = 0
    total_defects = 0
    confidence_sum = 0.0
    
    print("Running defect detection with analytics...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        marked_frame, defect_info = detector.process_frame(frame)
        frame_count += 1
        
        if defect_info:
            defect_type = defect_info['type']
            defect_counts[defect_type] += 1
            total_defects += 1
            confidence_sum += defect_info['confidence']
            
            # Alert if high confidence defect
            if defect_info['confidence'] > 0.85:
                print(f"⚠️  HIGH CONFIDENCE DEFECT: {defect_type} ({defect_info['confidence']:.2f})")
        
        # Display real-time stats every 100 frames
        if frame_count % 100 == 0:
            detection_rate = (total_defects / frame_count) * 100
            avg_confidence = confidence_sum / max(1, total_defects)
            
            print(f"\n--- Frame {frame_count} ---")
            print(f"Detection Rate: {detection_rate:.1f}%")
            print(f"Avg Confidence: {avg_confidence:.2f}")
            print(f"Defects by type: {dict(defect_counts)}")
        
        cv2.imshow("Detection", marked_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Final statistics
    print("\n=== INSPECTION COMPLETE ===")
    print(f"Total Frames: {frame_count}")
    print(f"Total Defects: {total_defects}")
    print(f"Overall Detection Rate: {(total_defects/frame_count)*100:.1f}%")
    print(f"Defect Breakdown:")
    for defect_type, count in sorted(defect_counts.items()):
        percentage = (count / total_defects) * 100
        print(f"  - {defect_type}: {count} ({percentage:.1f}%)")
    
    cap.release()
    cv2.destroyAllWindows()

# Run
run_with_analytics()


# ==============================================================================
# EXAMPLE 7: PLC Integration (Industrial Automation)
# ==============================================================================

from src.realtime_defect_detector import DefectDetector
from pymodbus.client import ModbusTcpClient
import cv2

def send_defect_signal_to_plc(defect_found, plc_ip="192.168.1.10", plc_port=502):
    """Send defect signal to PLC via Modbus TCP"""
    try:
        client = ModbusTcpClient(plc_ip, port=plc_port, timeout=1)
        if client.connect():
            # Write to register 1 (coil): True if defect found
            result = client.write_coil(1, defect_found)
            client.close()
            return result is not None
        return False
    except Exception as e:
        print(f"PLC Error: {e}")
        return False

def run_with_plc_integration():
    """Run detection with PLC signals"""
    detector = DefectDetector()
    cap = cv2.VideoCapture(0)
    
    print("Starting detection with PLC integration...")
    
    last_defect_state = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        marked_frame, defect_info = detector.process_frame(frame)
        
        # Determine if defect present and high confidence
        defect_found = (defect_info is not None and 
                       defect_info['confidence'] > 0.65)
        
        # Send signal to PLC if state changed
        if defect_found != last_defect_state:
            success = send_defect_signal_to_plc(defect_found)
            status = "✅" if success else "❌"
            print(f"{status} PLC Signal: Defect={defect_found}")
            last_defect_state = defect_found
        
        # Display
        status_text = "DEFECT!" if defect_found else "OK"
        color = (0, 0, 255) if defect_found else (0, 255, 0)
        cv2.putText(marked_frame, status_text, (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        
        cv2.imshow("Production Line", marked_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Ensure PLC is reset
    send_defect_signal_to_plc(False)
    
    cap.release()
    cv2.destroyAllWindows()

# Run with PLC
run_with_plc_integration()


# ==============================================================================
# EXAMPLE 8: Batch Processing Video Files
# ==============================================================================

from src.realtime_defect_detector import DefectDetector
import cv2
import os
import glob

def batch_process_videos(video_dir, output_dir="batch_results"):
    """Process multiple video files"""
    os.makedirs(output_dir, exist_ok=True)
    
    video_files = glob.glob(os.path.join(video_dir, "*.mp4")) + \
                  glob.glob(os.path.join(video_dir, "*.avi"))
    
    total_stats = {
        'files_processed': 0,
        'total_frames': 0,
        'total_defects': 0
    }
    
    for video_file in video_files:
        print(f"\nProcessing: {os.path.basename(video_file)}")
        
        detector = DefectDetector(
            log_file=os.path.join(output_dir, f"{os.path.basename(video_file)}.csv"),
            image_dir=os.path.join(output_dir, os.path.basename(video_file).split('.')[0])
        )
        
        cap = cv2.VideoCapture(video_file)
        frame_count = 0
        defect_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            marked_frame, defect_info = detector.process_frame(frame)
            frame_count += 1
            
            if defect_info:
                defect_count += 1
        
        cap.release()
        
        print(f"  Frames: {frame_count}, Defects: {defect_count}")
        
        total_stats['files_processed'] += 1
        total_stats['total_frames'] += frame_count
        total_stats['total_defects'] += defect_count
    
    # Summary
    print(f"\n=== BATCH PROCESSING COMPLETE ===")
    print(f"Files Processed: {total_stats['files_processed']}")
    print(f"Total Frames: {total_stats['total_frames']}")
    print(f"Total Defects: {total_stats['total_defects']}")

# Process all videos in a folder
batch_process_videos("./videos_to_inspect")


# ==============================================================================
# EXAMPLE 9: Export Defects for Manual Review
# ==============================================================================

from src.realtime_defect_detector import DefectLogAnalyzer
import webbrowser
import os

def create_html_gallery():
    """Create HTML gallery of detected defects"""
    analyzer = DefectLogAnalyzer()
    
    html = """
    <html>
    <head>
        <title>Defects Gallery</title>
        <style>
            body { font-family: Arial; margin: 20px; }
            .defect { display: inline-block; margin: 10px; border: 1px solid #ccc; padding: 10px; }
            img { width: 300px; }
            .info { margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1>Detected Defects</h1>
    """
    
    for _, row in analyzer.df.iterrows():
        if row['Image_Path'] and os.path.exists(row['Image_Path']):
            html += f"""
            <div class="defect">
                <img src="{row['Image_Path']}">
                <div class="info">
                    <b>{row['Defect_Type']}</b><br>
                    Confidence: {row['Confidence']}<br>
                    Time: {row['Timestamp']}
                </div>
            </div>
            """
    
    html += "</body></html>"
    
    # Save and open
    with open("defects_gallery.html", "w") as f:
        f.write(html)
    
    webbrowser.open("defects_gallery.html")

# Create gallery
create_html_gallery()


# ==============================================================================
# EXAMPLE 10: Continuous Monitoring with Logging
# ==============================================================================

import logging
from src.realtime_defect_detector import DefectDetector
import cv2

def setup_logging():
    """Setup comprehensive logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('defect_detection.log'),
            logging.StreamHandler()
        ]
    )

def run_continuous_monitoring(duration_minutes=480):
    """Run continuous monitoring (default 8 hours)"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting continuous defect monitoring")
    
    detector = DefectDetector()
    cap = cv2.VideoCapture(0)
    
    import time
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    frame_count = 0
    
    while time.time() < end_time:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame, reconnecting...")
            cap = cv2.VideoCapture(0)
            continue
        
        marked_frame, defect_info = detector.process_frame(frame)
        frame_count += 1
        
        if defect_info:
            logger.warning(
                f"DEFECT DETECTED: {defect_info['type']} "
                f"(Confidence: {defect_info['confidence']:.2f})"
            )
        
        if frame_count % 3600 == 0:  # Every hour
            hours_elapsed = (time.time() - start_time) / 3600
            logger.info(f"Status: {hours_elapsed:.1f} hours elapsed, {frame_count} frames processed")
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info("Monitoring stopped by user")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Continuous monitoring complete")

# Run for 8 hours
# run_continuous_monitoring(duration_minutes=480)


# ==============================================================================
# That's it! Pick an example and adapt to your needs.
# For more help, see IMPROVEMENT_GUIDE.md
# ==============================================================================
