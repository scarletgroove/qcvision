"""
Real-time Defect Detection for Paper Roll Inspection
Uses OpenCV-based computer vision techniques (no ML training required).
Detects: dark contamination, bright contamination, surface defects, wrinkles, color anomalies.
"""

import configparser
import csv
import logging
import os
import time
from datetime import datetime

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DefectDetector:
    """
    Real-time defect detection for paper roll inspection using OpenCV.
    No ML training required — uses computer vision techniques.

    Returns ALL defects per frame (not just the highest-confidence one),
    and de-duplicates log entries using a per-type cooldown timer.
    """

    DEFECT_TYPES = {
        0: "no_defect",
        1: "dark_contamination",
        2: "bright_contamination",
        3: "surface_defect",
        4: "wrinkle_or_fold",
        5: "color_anomaly",
    }

    # BGR colours for bounding-box overlays
    _DEFECT_COLORS = {
        1: (0,   0,   255),   # red
        2: (0,   165, 255),   # orange
        3: (0,   255, 255),   # yellow
        4: (255, 0,   255),   # magenta
        5: (255, 0,   0),     # blue
    }

    def __init__(self, log_file="defect_log.csv", image_dir="defective_images",
                 config_path="config.ini", save_images: bool = False):
        self.log_file    = log_file
        self.image_dir   = image_dir
        self.save_images = save_images
        self._load_config(config_path)
        self._initialize_logging()
        # Per-type timestamp of last CSV log entry (for cooldown)
        self._last_logged: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _load_config(self, config_path: str) -> None:
        config = configparser.ConfigParser()
        config.read(config_path)
        sec = "DETECTION_SETTINGS"
        self.DARK_THRESHOLD       = config.getint  (sec, "DARK_THRESHOLD",       fallback=50)
        self.BRIGHT_THRESHOLD     = config.getint  (sec, "BRIGHT_THRESHOLD",     fallback=200)
        self.MIN_DEFECT_AREA      = config.getint  (sec, "MIN_DEFECT_AREA",      fallback=20)
        self.MAX_DEFECT_AREA      = config.getint  (sec, "MAX_DEFECT_AREA",      fallback=5000)
        self.CONFIDENCE_THRESHOLD = config.getfloat(sec, "CONFIDENCE_THRESHOLD", fallback=0.5)
        self.LOG_COOLDOWN_SEC     = config.getfloat(sec, "LOG_COOLDOWN_SEC",     fallback=1.0)

    # ------------------------------------------------------------------
    # Logging / storage initialisation
    # ------------------------------------------------------------------

    def _initialize_logging(self) -> None:
        os.makedirs(self.image_dir, exist_ok=True)
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "Defect_Type", "Confidence",
                    "Image_Width", "Image_Height",
                    "X_Coord", "Y_Coord", "Bounding_Box",
                    "Area_Pixels", "Image_Path",
                ])
            logger.info("Created new defect log: %s", self.log_file)

    # ------------------------------------------------------------------
    # Detection methods
    # ------------------------------------------------------------------

    def detect_dark_particles(self, frame_gray: np.ndarray) -> list[dict]:
        """Detect dark foreign particles (contamination, dirt) on paper surface."""
        defects = []
        _, dark_mask = cv2.threshold(
            frame_gray, self.DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (self.MIN_DEFECT_AREA < area < self.MAX_DEFECT_AREA):
                continue
            x, y, w, h = cv2.boundingRect(contour)
            roi = frame_gray[y:y+h, x:x+w]
            mean_val = np.mean(roi)
            contrast  = np.std(roi)
            raw_conf = (self.DARK_THRESHOLD - mean_val) / max(1, self.DARK_THRESHOLD) * contrast / 50
            confidence = max(0.0, min(1.0, raw_conf))
            if confidence >= self.CONFIDENCE_THRESHOLD:
                defects.append({
                    "type": 1,
                    "bbox": (x, y, x+w, y+h),
                    "confidence": confidence,
                    "area": area,
                })
        return defects

    def detect_bright_particles(self, frame_gray: np.ndarray) -> list[dict]:
        """Detect bright foreign particles (dust, reflections) on paper surface."""
        defects = []
        _, bright_mask = cv2.threshold(
            frame_gray, self.BRIGHT_THRESHOLD, 255, cv2.THRESH_BINARY
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (self.MIN_DEFECT_AREA < area < self.MAX_DEFECT_AREA):
                continue
            x, y, w, h = cv2.boundingRect(contour)
            roi = frame_gray[y:y+h, x:x+w]
            mean_val = np.mean(roi)
            contrast  = np.std(roi)
            denom = max(1, 255 - self.BRIGHT_THRESHOLD)
            raw_conf = (mean_val - self.BRIGHT_THRESHOLD) / denom * contrast / 50
            confidence = max(0.0, min(1.0, raw_conf))
            if confidence >= self.CONFIDENCE_THRESHOLD:
                defects.append({
                    "type": 2,
                    "bbox": (x, y, x+w, y+h),
                    "confidence": confidence,
                    "area": area,
                })
        return defects

    def detect_surface_defects(self, frame_gray: np.ndarray) -> list[dict]:
        """Detect scratches and surface irregularities using Canny edge detection."""
        defects = []
        blurred = cv2.GaussianBlur(frame_gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, 50, 150)
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges   = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_min = self.MIN_DEFECT_AREA * 2
        area_max = self.MAX_DEFECT_AREA * 2
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (area_min < area < area_max):
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / max(1, h)
            if not (0.1 < aspect_ratio < 10):
                continue
            roi_edges  = edges[y:y+h, x:x+w]
            edge_density = np.count_nonzero(roi_edges) / max(1, area)
            confidence = max(0.0, min(1.0, edge_density * 5))
            if confidence >= self.CONFIDENCE_THRESHOLD * 0.7:
                defects.append({
                    "type": 3,
                    "bbox": (x, y, x+w, y+h),
                    "confidence": confidence,
                    "area": area,
                })
        return defects

    def detect_wrinkles(self, frame_gray: np.ndarray) -> list[dict]:
        """Detect wrinkles and folds using Laplacian variance."""
        defects = []
        laplacian     = cv2.Laplacian(frame_gray, cv2.CV_64F)
        laplacian_abs = np.absolute(laplacian)
        lap_max       = laplacian_abs.max()
        if lap_max == 0:
            return defects
        laplacian_u8 = np.uint8(laplacian_abs / lap_max * 255)
        _, wrinkle_mask = cv2.threshold(laplacian_u8, 100, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        wrinkle_mask = cv2.morphologyEx(wrinkle_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(wrinkle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_min = self.MIN_DEFECT_AREA * 3
        area_max = self.MAX_DEFECT_AREA * 4
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (area_min < area < area_max):
                continue
            if len(contour) < 5:
                continue
            _, (minor, major), _ = cv2.fitEllipse(contour)
            if major == 0 or (minor / major) >= 0.4:
                continue  # Not elongated enough to be a wrinkle
            x, y, w, h = cv2.boundingRect(contour)
            roi_lap    = laplacian_u8[y:y+h, x:x+w]
            confidence = max(0.0, min(1.0, float(np.mean(roi_lap)) / 255))
            if confidence >= self.CONFIDENCE_THRESHOLD * 0.5:
                defects.append({
                    "type": 4,
                    "bbox": (x, y, x+w, y+h),
                    "confidence": confidence,
                    "area": area,
                })
        return defects

    def detect_color_anomalies(self, frame: np.ndarray) -> list[dict]:
        """Detect colour/shade anomalies using HSV colour space."""
        defects = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        mask_low_sat = cv2.inRange(s, 0,   50)
        mask_low_val = cv2.inRange(v, 0,  100)
        color_mask   = cv2.bitwise_or(mask_low_sat, mask_low_val)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_min = self.MIN_DEFECT_AREA * 2
        area_max = self.MAX_DEFECT_AREA * 3
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (area_min < area < area_max):
                continue
            x, y, w, h = cv2.boundingRect(contour)
            roi_s      = s[y:y+h, x:x+w]
            sat_mean   = np.mean(roi_s)
            confidence = max(0.0, min(1.0, abs(sat_mean - 128) / 128))
            if confidence >= self.CONFIDENCE_THRESHOLD * 0.6:
                defects.append({
                    "type": 5,
                    "bbox": (x, y, x+w, y+h),
                    "confidence": confidence,
                    "area": area,
                })
        return defects

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def detect_defects(self, frame: np.ndarray) -> list[dict]:
        """Run all detection methods and return every defect found."""
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        all_defects: list[dict] = []
        all_defects.extend(self.detect_dark_particles(frame_gray))
        all_defects.extend(self.detect_bright_particles(frame_gray))
        all_defects.extend(self.detect_surface_defects(frame_gray))
        all_defects.extend(self.detect_color_anomalies(frame))
        all_defects.extend(self.detect_wrinkles(frame_gray))
        return all_defects

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_defect(self, frame: np.ndarray, defect: dict) -> None:
        x1, y1, x2, y2 = defect["bbox"]
        color      = self._DEFECT_COLORS.get(defect["type"], (0, 0, 255))
        label      = self.DEFECT_TYPES.get(defect["type"], "unknown")
        conf       = defect["confidence"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text_y = y1 - 10 if y1 > 20 else y2 + 20
        cv2.putText(frame, f"{label} ({conf:.2f})",
                    (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # ------------------------------------------------------------------
    # Logging / persistence
    # ------------------------------------------------------------------

    def _should_log(self, defect_type_id: int) -> bool:
        """Return True if the cooldown period has elapsed for this defect type."""
        now  = time.monotonic()
        last = self._last_logged.get(defect_type_id, 0.0)
        if now - last >= self.LOG_COOLDOWN_SEC:
            self._last_logged[defect_type_id] = now
            return True
        return False

    def _log_defect_row(self, defect: dict, frame: np.ndarray, image_path: str) -> None:
        """Append one row to the CSV log file."""
        x1, y1, x2, y2 = defect["bbox"]
        h, w = frame.shape[:2]
        timestamp     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        defect_name   = self.DEFECT_TYPES.get(defect["type"], "unknown")
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                defect_name,
                f"{defect['confidence']:.2f}",
                w, h,
                (x1 + x2) // 2,
                (y1 + y2) // 2,
                f"{x1},{y1},{x2},{y2}",
                defect["area"],
                image_path,
            ])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """
        Process a single frame.

        Returns
        -------
        marked_frame : np.ndarray
            Copy of the frame with bounding boxes drawn for all detected defects.
        defect_infos : list[dict]
            One entry per detected defect with keys: type, confidence, bbox, area.
            Empty list when no defects are found.
        """
        defects = self.detect_defects(frame)
        frame_marked = frame.copy()
        defect_infos: list[dict] = []

        if not defects:
            return frame_marked, defect_infos

        # Draw all bounding boxes on a single marked frame
        for defect in defects:
            self._draw_defect(frame_marked, defect)

        # Optionally save one annotated image per detection event
        image_path = ""
        if self.save_images:
            timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            image_filename = f"defect_frame_{timestamp_file}.jpg"
            image_path     = os.path.join(self.image_dir, image_filename)
            cv2.imwrite(image_path, frame_marked)

        for defect in defects:
            defect_infos.append({
                "type":       self.DEFECT_TYPES[defect["type"]],
                "confidence": defect["confidence"],
                "bbox":       defect["bbox"],
                "area":       defect["area"],
            })
            # Log to CSV only if cooldown allows for this defect type
            if self._should_log(defect["type"]):
                self._log_defect_row(defect, frame, image_path)
                logger.info(
                    "Defect logged: %s conf=%.2f bbox=%s",
                    defect_infos[-1]["type"], defect["confidence"], defect["bbox"]
                )

        return frame_marked, defect_infos


# ---------------------------------------------------------------------------
# Stand-alone CLI runner
# ---------------------------------------------------------------------------

def run_realtime_detection(video_source=0, detection_threshold=0.5) -> None:
    """
    Run real-time defect detection in a cv2 window.

    Parameters
    ----------
    video_source : int | str
        0 for webcam, path to video file, or RTSP URL.
    detection_threshold : float
        Confidence threshold for logging defects.
    """
    detector = DefectDetector()
    detector.CONFIDENCE_THRESHOLD = detection_threshold

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error("Failed to open video source: %s", video_source)
        return

    logger.info("Starting real-time defect detection — press Q to quit, S to save frame")
    frame_count  = 0
    defect_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video or error reading frame")
                break

            frame_count += 1
            marked_frame, defect_infos = detector.process_frame(frame)

            if defect_infos:
                defect_count += 1

            # Overlay stats
            display = marked_frame.copy()
            cv2.putText(display, f"Frame: {frame_count} | Defect events: {defect_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if defect_infos:
                summary = ", ".join(
                    f"{d['type']}({d['confidence']:.2f})" for d in defect_infos
                )
                cv2.putText(display, summary,
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow("Real-time Defect Detection", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                path = f"debug_frame_{frame_count}.jpg"
                cv2.imwrite(path, display)
                logger.info("Frame saved: %s", path)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info(
            "Detection complete — %d frames processed, %d defect events logged",
            frame_count, defect_count,
        )


if __name__ == "__main__":
    run_realtime_detection(video_source=0, detection_threshold=0.5)
