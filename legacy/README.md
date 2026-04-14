# Legacy — TensorFlow-based implementation (v1)

These files are the original v1 implementation that used TensorFlow/Keras placeholder ML models.
They are **not actively maintained** and require `tensorflow` which is not in `requirements.txt`.

## Files

- `app.py` — old Streamlit entry point (imports from `src/main.py`)
- `src/main.py` — TF model creation, training stubs, PLC process workers
- `src/video_processing.py` — `VideoProcessor` wrapper around the TF-based pipeline

## Active codebase

Use these instead:

| Purpose | File |
|---|---|
| Web UI | `streamlit_app.py` |
| Detection engine | `src/realtime_defect_detector.py` |
| Calibration | `calibrate_detector.py` |
| Log analysis | `analyze_defects.py` |
