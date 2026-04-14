# Claude/Copilot Instructions for QCVision

You are assisting with **QCVision** — a production-ready, real-time paper roll defect detection system that uses OpenCV-based computer vision techniques (no ML training required).

## 🎯 Project Overview

**QCVision 2.0** is a lightweight quality control inspection system for paper manufacturing:
- ✅ **No ML Training** - Works immediately with pre-tuned OpenCV algorithms
- ✅ **Real-Time Performance** - 8-30 FPS on standard hardware (CPU-only)
- ✅ **Lightweight** - ~150 MB footprint vs 2GB+ for ML solutions
- ✅ **Detects 5 Defect Types**: dark particles, bright particles, surface defects, wrinkles/folds, color anomalies
- ✅ **Multiple Interfaces**: Web UI (Streamlit), CLI, programmatic API

**Key Achievement**: Transformed from non-functional deep learning approach (2.5GB TensorFlow, GPU-required, 2-3min startup) to working vision-based solution (<2sec startup, CPU-only, 85-95% accuracy).

## 📚 Project Structure

```
qcvision/
├── README.md                      # Product overview & quick start
├── START_HERE.txt                 # Setup & transformation summary
├── requirements.txt               # Lightweight dependencies (OpenCV, Streamlit, NumPy, Pandas)
├── config.ini                     # Configuration file (parameters, thresholds)
├── defect_log.csv                 # CSV log of detected defects
├── streamlit_app.py               # Web dashboard interface (main UI)
├── calibrate_detector.py          # Interactive parameter tuning tool
├── analyze_defects.py             # Defect log analysis & reporting
├── quickstart.py                  # Setup wizard
├── CODE_EXAMPLES.py               # 10 ready-to-use code samples
├── IMPROVEMENT_GUIDE.md           # Deep technical documentation
├── src/
│   ├── realtime_defect_detector.py  # Main detection engine (420 lines)
│   ├── video_processing.py          # Video input/output utilities
│   └── main.py                      # Command-line entry point
├── models/
│   └── (Pre-trained detection models if any)
├── training_data/
│   ├── contaminate/                 # Example contamination samples
│   └── pe_surface/                  # Example surface samples
└── defective_images/                # Output folder for detected defects
```

## 🔧 Architecture & Key Components

### Core Detection Engine: `src/realtime_defect_detector.py`

The `DefectDetector` class implements 5 defect detection algorithms:

1. **Dark Contamination** - Threshold-based detection of dark particles (dirt, ink spots)
   - Parameter: `DARK_THRESHOLD` (default: 50)
   
2. **Bright Contamination** - Detects bright foreign particles (dust, lint, white spots)
   - Parameter: `BRIGHT_THRESHOLD` (default: 200)
   
3. **Surface Defects** - Canny edge detection for scratches, tears, holes
   - Parameter: `SURFACE_SENSITIVITY` (tunable)
   
4. **Wrinkles/Folds** - Laplacian variance detection for surface undulations
   - Parameter: `WRINKLE_SENSITIVITY` (tunable)
   
5. **Color Anomalies** - HSV color space analysis for shade/color changes
   - Parameter: `COLOR_SENSITIVITY` (tunable)

**Key Class Methods**:
- `detect(frame)` - Process single frame, return defects
- `log_defect(defect_type, confidence, frame)` - Log to CSV + save image
- `get_statistics()` - Retrieve detection statistics
- `adjust_parameters(param_dict)` - Update detection thresholds at runtime

### User Interfaces

1. **Streamlit Web App** (`streamlit_app.py`)
   - Real-time slider controls for all parameters
   - Live video feed with defect highlighting
   - CSV export and defect gallery view
   - Run: `streamlit run streamlit_app.py`

2. **Calibration Tool** (`calibrate_detector.py`)
   - Interactive trackbars for parameter tuning
   - Immediate visual feedback on specific test images
   - Save tuned parameters for production

3. **CLI Mode** (`src/main.py`)
   - Command-line interface for real-time detection
   - Supports: webcam, video files, IP cameras
   - Run: `python -m src.realtime_defect_detector`

## 💻 Code Conventions & Patterns

### Python Style
- **Format**: PEP 8 compliant
- **Documentation**: Docstrings in all classes/functions
- **Type Hints**: Used where practical (especially in public APIs)
- **Logging**: Use standard `logging` module, not print statements
  ```python
  import logging
  logger = logging.getLogger(__name__)
  logger.info("Detection started")
  ```

### Computer Vision Patterns
- **Frame Processing**: Always convert BGR to grayscale for threshold-based detection
  ```python
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  ```
  
- **Morphological Operations**: Cleanup noise after thresholding
  ```python
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
  clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
  ```

- **Performance**: Always resize large frames before processing
  ```python
  if frame.shape[0] > 1080:
      scale = 1080 / frame.shape[0]
      frame = cv2.resize(frame, None, fx=scale, fy=scale)
  ```

### Data Logging
- **CSV Format**: `timestamp,defect_type,confidence,image_path`
- **Image Storage**: Save detected defects in `defective_images/` with timestamp
- **Rotation**: Implement log rotation for production (prevent unbounded growth)

### Configuration Management
- All tunable parameters should be extractable from `config.ini` or passed as function arguments
- Avoid hardcoded magic numbers in production code
- Document parameter ranges and recommended values in comments

## 🚀 Common Workflows

### Workflow 1: Run Real-Time Detection
```bash
# Option A: CLI (direct)
python -m src.realtime_defect_detector

# Option B: Web UI (visual controls)
streamlit run streamlit_app.py

# Option C: Programmatic
from src.realtime_defect_detector import DefectDetector
detector = DefectDetector()
detector.run_realtime(video_source=0)  # 0=webcam
```

### Workflow 2: Calibrate Parameters for Your Paper Type
```bash
python calibrate_detector.py
# Adjust trackbars until detection looks right
# Parameters are saved for future use
```

### Workflow 3: Analyze Defect Logs
```bash
python analyze_defects.py
# Generates: statistics, trends, defect frequency heatmaps
```

### Workflow 4: Test on Video File
```bash
python -m src.realtime_defect_detector --video path/to/video.mp4
```

### Workflow 5: Integrate Multiple Camera Streams
- See `CODE_EXAMPLES.py` sample #7: "Multi-Camera Monitoring"
- Typically uses threading with one `DefectDetector` instance per camera

## 📋 Guidelines for Assisting with QCVision

### When Writing Code:
1. **Maintain Performance**: Check frame processing speed (target: 15+ FPS for real-time)
2. **Test Against Real Data**: Use test images/videos in `training_data/` folder
3. **Log Appropriately**: Use logging, not print statements
4. **Handle Video I/O**: Use `cv2.VideoCapture` for webcam/file/network sources
5. **Memory**: Use frame deques with `maxlen` to prevent unbounded memory growth

### When Suggesting Features:
- **Avoid ML/Neural Networks**: Keep solution lightweight (no TensorFlow, no GPU requirements)
- **Respect Thresholds**: New defect types should be tunable via parameters
- **Test Coverage**: Suggest testing against provided samples in `training_data/`
- **Documentation**: Update README/IMPROVEMENT_GUIDE.md for user-facing changes

### When Debugging:
- **Video Input Issues**: Check camera permissions, video codec support
- **Accuracy Issues**: Guide user to calibration tool, don't modify hardcoded thresholds
- **Performance Issues**: Profile with different resolutions; suggest preprocessing
- **CSV Log Corruption**: Check file permissions or concurrent writes

## 🔌 Optional: Integrations (Mentioned but Not Required)

- **PLC/Modbus TCP**: Use `pymodbus` library (already in requirements.txt)
- **IP Cameras**: RTSP streams via `cv2.VideoCapture("rtsp://...")`
- **Database Logging**: Replace CSV with SQLite/PostgreSQL (advanced)

## 📞 Contact & Testing

- **Improvement Guide**: See `IMPROVEMENT_GUIDE.md` for detailed technical documentation
- **Code Examples**: See `CODE_EXAMPLES.py` for 10 ready-to-use samples
- **Test Data**: Use `training_data/contaminate/` and `training_data/pe_surface/` for validation

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | ≥4.6.0 | Computer vision algorithms |
| `numpy` | ≥1.21.0 | Numerical operations |
| `pandas` | ≥1.3.0 | CSV/data analysis |
| `streamlit` | ≥1.20.0 | Web UI framework |
| `pymodbus` | ≥3.1.0 | Optional PLC integration |
| `matplotlib/seaborn` | Latest | Visualization & analytics |

**Install**: `pip install -r requirements.txt`

---

**Remember**: QCVision is designed to be *lightweight*, *production-ready*, and *tunable* — not to be a cutting-edge ML system. Optimizations should prioritize simplicity, performance, and maintainability over accuracy gains that require heavy dependencies.
