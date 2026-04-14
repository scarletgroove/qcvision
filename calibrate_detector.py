#!/usr/bin/env python3
"""
Interactive Calibration Tool for Defect Detection
Helps find optimal parameters for your specific paper type
"""

import cv2
import numpy as np
from src.realtime_defect_detector import DefectDetector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CalibrationTool:
    """Interactive calibration with trackbars"""
    
    def __init__(self, video_source=0):
        self.detector = DefectDetector()
        self.video_source = video_source
        self.cap = cv2.VideoCapture(video_source)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {video_source}")
            return
        
        # Create window
        self.window_name = "Defect Detection Calibration Tool"
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        
        # Create trackbars
        self._create_trackbars()
    
    def _create_trackbars(self):
        """Create all trackbar controls"""
        
        # Confidence threshold
        cv2.createTrackbar('Confidence Threshold', self.window_name, 
                          int(self.detector.CONFIDENCE_THRESHOLD * 100), 100,
                          lambda x: setattr(self.detector, 'CONFIDENCE_THRESHOLD', x/100.0))
        
        # Dark threshold
        cv2.createTrackbar('Dark Threshold', self.window_name,
                          self.detector.DARK_THRESHOLD, 255,
                          lambda x: setattr(self.detector, 'DARK_THRESHOLD', x))
        
        # Bright threshold
        cv2.createTrackbar('Bright Threshold', self.window_name,
                          self.detector.BRIGHT_THRESHOLD, 255,
                          lambda x: setattr(self.detector, 'BRIGHT_THRESHOLD', x))
        
        # Min area
        cv2.createTrackbar('Min Area', self.window_name,
                          self.detector.MIN_DEFECT_AREA, 200,
                          lambda x: setattr(self.detector, 'MIN_DEFECT_AREA', max(1, x)))
        
        # Max area
        cv2.createTrackbar('Max Area', self.window_name,
                          self.detector.MAX_DEFECT_AREA // 50, 200,
                          lambda x: setattr(self.detector, 'MAX_DEFECT_AREA', max(100, x * 50)))
    
    def run(self):
        """Run interactive calibration"""
        logger.info("Starting calibration tool...")
        logger.info("Controls:")
        logger.info("  SPACE - Freeze frame for inspection")
        logger.info("  S     - Save current frame and parameters")
        logger.info("  C     - Copy parameters to clipboard (or print)")
        logger.info("  R     - Reset to default parameters")
        logger.info("  Q     - Quit")
        
        total_frames = 0
        defected_frames = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.info("End of video stream")
                    break
                
                # Process frame — returns list of defects (may be empty)
                marked_frame, defect_infos = self.detector.process_frame(frame)
                total_frames += 1

                if defect_infos:
                    defected_frames += 1
                
                # Add calibration info overlay
                h, w = frame.shape[:2]
                
                # Draw information panel
                overlay = marked_frame.copy()
                cv2.rectangle(overlay, (10, 10), (500, 300), (0, 0, 0), -1)
                marked_frame = cv2.addWeighted(overlay, 0.3, marked_frame, 0.7, 0)
                
                # Parameters text
                text_y = 35
                cv2.putText(marked_frame, f"CALIBRATION MODE", (20, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                text_y += 40
                cv2.putText(marked_frame, f"Confidence: {self.detector.CONFIDENCE_THRESHOLD:.2f}",
                           (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                text_y += 30
                cv2.putText(marked_frame, f"Dark Threshold: {self.detector.DARK_THRESHOLD}",
                           (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                text_y += 30
                cv2.putText(marked_frame, f"Bright Threshold: {self.detector.BRIGHT_THRESHOLD}",
                           (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                text_y += 30
                cv2.putText(marked_frame, f"Min Area: {self.detector.MIN_DEFECT_AREA}  Max Area: {self.detector.MAX_DEFECT_AREA}",
                           (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                text_y += 40
                detection_rate = (defected_frames / max(1, total_frames)) * 100
                cv2.putText(marked_frame, f"Frames: {total_frames} | Defects: {defected_frames} ({detection_rate:.1f}%)",
                           (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                if defect_infos:
                    text_y += 30
                    summary = ", ".join(
                        f"{d['type']}({d['confidence']:.2f})" for d in defect_infos
                    )
                    cv2.putText(marked_frame, summary,
                               (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                
                # Display
                cv2.imshow(self.window_name, marked_frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_frame(marked_frame)
                elif key == ord('c'):
                    self._copy_parameters()
                elif key == ord('r'):
                    self._reset_parameters()
                elif key == ord(' '):
                    self._pause_for_inspection(marked_frame)
        
        finally:
            self.cleanup()
    
    def _save_frame(self, frame):
        """Save current frame and parameters"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save frame
        frame_path = f"calibration_frame_{timestamp}.jpg"
        cv2.imwrite(frame_path, frame)
        
        # Save parameters
        params_path = f"calibration_params_{timestamp}.txt"
        with open(params_path, 'w') as f:
            f.write(f"Calibration Parameters - {timestamp}\n")
            f.write(f"================================\n\n")
            f.write(f"CONFIDENCE_THRESHOLD = {self.detector.CONFIDENCE_THRESHOLD}\n")
            f.write(f"DARK_THRESHOLD = {self.detector.DARK_THRESHOLD}\n")
            f.write(f"BRIGHT_THRESHOLD = {self.detector.BRIGHT_THRESHOLD}\n")
            f.write(f"MIN_DEFECT_AREA = {self.detector.MIN_DEFECT_AREA}\n")
            f.write(f"MAX_DEFECT_AREA = {self.detector.MAX_DEFECT_AREA}\n")
        
        logger.info(f"Saved: {frame_path}, {params_path}")
    
    def _copy_parameters(self):
        """Print current parameters as code"""
        print("\n" + "="*50)
        print("COPY THESE PARAMETERS TO YOUR CODE:")
        print("="*50)
        print(f"\ndetector.CONFIDENCE_THRESHOLD = {self.detector.CONFIDENCE_THRESHOLD}")
        print(f"detector.DARK_THRESHOLD = {self.detector.DARK_THRESHOLD}")
        print(f"detector.BRIGHT_THRESHOLD = {self.detector.BRIGHT_THRESHOLD}")
        print(f"detector.MIN_DEFECT_AREA = {self.detector.MIN_DEFECT_AREA}")
        print(f"detector.MAX_DEFECT_AREA = {self.detector.MAX_DEFECT_AREA}")
        print("\n")
    
    def _reset_parameters(self):
        """Reset to default parameters"""
        self.detector = DefectDetector()
        self._create_trackbars()
        logger.info("Parameters reset to defaults")
    
    def _pause_for_inspection(self, frame):
        """Pause for detailed inspection"""
        logger.info("Paused. Press any key to continue...")
        cv2.imshow(self.window_name, frame)
        cv2.waitKey(0)
    
    def cleanup(self):
        """Cleanup resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Calibration complete!")


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("DEFECT DETECTION CALIBRATION TOOL")
    print("="*60)
    print("\nThis tool helps you find optimal parameters for your paper type.")
    print("Use the trackbars to adjust sensitivity and thresholds.")
    print("\nPress SPACE to pause, S to save, C to copy parameters, Q to quit\n")
    
    # Choose video source
    sources = {
        "1": {"name": "Webcam", "source": 0},
        "2": {"name": "Video File", "source": None},
        "3": {"name": "IP Camera", "source": None}
    }
    
    print("Available Video Sources:")
    for key, source in sources.items():
        print(f"  [{key}] {source['name']}")
    
    choice = input("\nSelect source (1-3): ").strip()
    
    if choice == "1":
        source = 0
    elif choice == "2":
        path = input("Enter video file path: ").strip()
        source = path
    elif choice == "3":
        url = input("Enter IP camera URL (rtsp://...): ").strip()
        source = url
    else:
        print("Invalid choice")
        return
    
    # Run calibration
    tool = CalibrationTool(video_source=source)
    tool.run()


if __name__ == "__main__":
    main()
