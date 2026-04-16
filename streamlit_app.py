"""
QCVision — Paper Roll Defect Detection
Dark industrial dashboard UI with English / Thai language support.

Architecture:
  - Background thread: VideoCapture + DefectDetector → frame_queue
  - Main thread: reads queue, renders base64 HTML frames (bypasses Streamlit
    media store, fixing MediaFileStorageError on rapid reruns)
  - Stop button works via threading.Event checked between frames
"""

import base64
import glob as glob_module
import os
import queue as queue_module
import threading
import time
from collections import defaultdict
from datetime import datetime

import cv2
import pandas as pd
import streamlit as st

from src.realtime_defect_detector import DefectDetector

os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

# ---------------------------------------------------------------------------
# Translations
# ---------------------------------------------------------------------------

TRANSLATIONS: dict[str, dict[str, str]] = {
    # ── Sidebar ──────────────────────────────────────────────────────────
    "language_label":           {"en": "Language",              "th": "ภาษา"},
    "sidebar_title":            {"en": "QCVision",              "th": "QCVision"},
    "section_source":           {"en": "Video Source",          "th": "แหล่งวิดีโอ"},
    "source_webcam":            {"en": "Webcam",                "th": "เว็บแคม"},
    "source_file":              {"en": "Video File",            "th": "ไฟล์วิดีโอ"},
    "source_ip":                {"en": "IP Camera",             "th": "กล้อง IP"},
    "source_mjpeg":             {"en": "HTTP MJPEG",            "th": "HTTP MJPEG"},
    "source_images":            {"en": "Image Sequence",        "th": "ลำดับภาพ"},
    "source_gige":              {"en": "GigE Vision",           "th": "GigE Vision"},
    "webcam_index":             {"en": "Webcam Index",          "th": "หมายเลขเว็บแคม"},
    "upload_label":             {"en": "Video File",            "th": "เลือกไฟล์วิดีโอ"},
    "rtsp_label":               {"en": "RTSP URL",              "th": "URL กล้อง RTSP"},
    "rtsp_placeholder":         {"en": "rtsp://user:pass@ip/stream", "th": "rtsp://user:pass@ip/stream"},
    "mjpeg_label":              {"en": "MJPEG Stream URL",      "th": "URL สตรีม MJPEG"},
    "mjpeg_placeholder":        {"en": "http://192.168.1.x/video.mjpg", "th": "http://192.168.1.x/video.mjpg"},
    "images_label":             {"en": "Image Folder Path",     "th": "พาธโฟลเดอร์ภาพ"},
    "images_placeholder":       {"en": "/path/to/frames/",      "th": "/path/to/frames/"},
    "images_fps":               {"en": "Playback FPS",          "th": "FPS การเล่น"},
    "images_help":              {"en": "Folder containing .jpg / .png frames processed in filename order.",
                                 "th": "โฟลเดอร์ที่มีเฟรม .jpg / .png ประมวลผลตามลำดับชื่อไฟล์"},
    "gige_cti_label":           {"en": "CTI File Path",         "th": "พาธไฟล์ CTI"},
    "gige_cti_placeholder":    {"en": "/path/to/producer.cti", "th": "/path/to/producer.cti"},
    "gige_cti_help":           {"en": "Path to the GenTL producer (.cti) file supplied by your camera manufacturer.",
                                 "th": "พาธไปยังไฟล์ GenTL producer (.cti) ที่ผู้ผลิตกล้องจัดให้"},
    "gige_index":              {"en": "Camera Index",           "th": "หมายเลขกล้อง"},
    "gige_index_help":         {"en": "Index of the camera to open when multiple devices are detected.",
                                 "th": "หมายเลขกล้องที่ต้องการเปิดเมื่อพบอุปกรณ์หลายตัว"},
    "gige_help":               {"en": "Requires the 'harvesters' package:",
                                 "th": "ต้องติดตั้งแพ็กเกจ 'harvesters':"},
    "section_detection":        {"en": "Detection Parameters",  "th": "ค่าพารามิเตอร์การตรวจจับ"},
    "conf_threshold":           {"en": "Confidence Threshold",  "th": "ค่าความเชื่อมั่น"},
    "conf_help":                {"en": "Min confidence to register a detection.",
                                 "th": "ความเชื่อมั่นขั้นต่ำในการลงทะเบียนการตรวจจับ"},
    "dark_threshold":           {"en": "Dark Pixel Threshold",  "th": "ค่าขีดแบ่งพิกเซลสีเข้ม"},
    "dark_help":                {"en": "Pixels darker than this are treated as potential contamination.",
                                 "th": "พิกเซลที่เข้มกว่าค่านี้จะถูกพิจารณาว่าเป็นสิ่งปนเปื้อน"},
    "bright_threshold":         {"en": "Bright Pixel Threshold","th": "ค่าขีดแบ่งพิกเซลสีสว่าง"},
    "area_filters":             {"en": "Area Filters",          "th": "ตัวกรองพื้นที่"},
    "min_area":                 {"en": "Min Area (px²)",        "th": "พื้นที่ขั้นต่ำ (px²)"},
    "max_area":                 {"en": "Max Area (px²)",        "th": "พื้นที่สูงสุด (px²)"},
    "section_display":          {"en": "Display",               "th": "การแสดงผล"},
    "feed_height":              {"en": "Feed Height (px)",      "th": "ความสูงภาพ (px)"},
    # ── Header ───────────────────────────────────────────────────────────
    "app_subtitle":             {"en": "Real-time paper roll quality inspection",
                                 "th": "การตรวจสอบคุณภาพม้วนกระดาษแบบเรียลไทม์"},
    # ── Status badges ────────────────────────────────────────────────────
    "status_live":              {"en": "LIVE",                  "th": "กำลังทำงาน"},
    "status_stopped":           {"en": "STOPPED",               "th": "หยุดแล้ว"},
    # ── Buttons ──────────────────────────────────────────────────────────
    "btn_start":                {"en": "▶  Start",              "th": "▶  เริ่ม"},
    "btn_stop":                 {"en": "■  Stop",               "th": "■  หยุด"},
    "btn_clear":                {"en": "↺  Clear Log",          "th": "↺  ล้างบันทึก"},
    # ── Feed placeholder ─────────────────────────────────────────────────
    "feed_idle_title":          {"en": "No feed active",        "th": "ไม่มีสัญญาณวิดีโอ"},
    "feed_idle_sub":            {"en": "Select a source and press Start",
                                 "th": "เลือกแหล่งและกด เริ่ม"},
    "feed_ended":               {"en": "Stream ended.",         "th": "สัญญาณวิดีโอสิ้นสุดแล้ว"},
    "feed_no_source":           {"en": "Could not open source. Camera / device not found or inaccessible.",
                                 "th": "ไม่สามารถเปิดแหล่งวิดีโอได้ ไม่พบกล้องหรืออุปกรณ์"},
    "local_only_note":          {"en": "⚠️ Requires local deployment — not available on cloud.",
                                 "th": "⚠️ ใช้งานได้เฉพาะเมื่อติดตั้งในเครื่อง ไม่รองรับบนคลาวด์"},
    "no_source_warning":        {"en": "Configure a video source first.",
                                 "th": "โปรดตั้งค่าแหล่งวิดีโอก่อน"},
    # ── Metrics ──────────────────────────────────────────────────────────
    "metric_frames":            {"en": "Frames",                "th": "เฟรม"},
    "metric_events":            {"en": "Events",                "th": "เหตุการณ์"},
    "metric_rate":              {"en": "Rate",                  "th": "อัตรา"},
    # ── Stats cards ──────────────────────────────────────────────────────
    "card_by_type":             {"en": "By Type (session)",     "th": "ตามประเภท (เซสชัน)"},
    "card_recent":              {"en": "Recent Detections",     "th": "การตรวจจับล่าสุด"},
    # ── Defect type names ─────────────────────────────────────────────────
    "defect_1":                 {"en": "Dark Contamination",    "th": "สิ่งปนเปื้อนสีเข้ม"},
    "defect_2":                 {"en": "Bright Contamination",  "th": "สิ่งปนเปื้อนสีสว่าง"},
    "defect_3":                 {"en": "Surface Defect",        "th": "ข้อบกพร่องบนพื้นผิว"},
    "defect_4":                 {"en": "Wrinkle / Fold",        "th": "รอยย่น / รอยพับ"},
    "defect_5":                 {"en": "Color Anomaly",         "th": "ความผิดปกติของสี"},
    "defect_clear":             {"en": "✓  Clear",              "th": "✓  ปกติ"},
    # ── Defect log section ───────────────────────────────────────────────
    "log_expander":             {"en": "Defect Log",            "th": "บันทึกข้อบกพร่อง"},
    "log_total":                {"en": "Total Entries",         "th": "จำนวนรายการทั้งหมด"},
    "log_avg_conf":             {"en": "Avg Confidence",        "th": "ความเชื่อมั่นเฉลี่ย"},
    "log_types":                {"en": "Types Seen",            "th": "ประเภทที่พบ"},
    "log_empty":                {"en": "No entries yet.",       "th": "ยังไม่มีรายการ"},
    "log_no_file":              {"en": "Start detection to generate a log.",
                                 "th": "เริ่มการตรวจจับเพื่อสร้างบันทึก"},
    "log_download":             {"en": "Download CSV",          "th": "ดาวน์โหลด CSV"},
    "log_error":                {"en": "Could not read log:",   "th": "ไม่สามารถอ่านบันทึก:"},
    # ── Image gallery ─────────────────────────────────────────────────────
    "gallery_expander":         {"en": "Defective Images",      "th": "ภาพข้อบกพร่อง"},
    "gallery_empty":            {"en": "No defective images captured yet.",
                                 "th": "ยังไม่มีภาพข้อบกพร่องที่บันทึกไว้"},
    "gallery_count":            {"en": "Saved Images",          "th": "ภาพที่บันทึก"},
    # ── Image saving toggle ───────────────────────────────────────────────
    "save_images_toggle":       {"en": "Save Defect Images",    "th": "บันทึกภาพข้อบกพร่อง"},
    "save_images_help":         {"en": "Save annotated frames to the defective_images/ folder.",
                                 "th": "บันทึกเฟรมที่มีข้อบกพร่องลงในโฟลเดอร์ defective_images/"},
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="QCVision",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — dark industrial theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d1117;
    color: #e6edf3;
}
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label {
    color: #8b949e !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Lang switcher ── */
.lang-bar {
    display: flex;
    gap: 6px;
    margin-bottom: 16px;
}
.lang-btn {
    flex: 1;
    padding: 5px 0;
    border-radius: 6px;
    border: 1px solid #30363d;
    background: transparent;
    color: #8b949e;
    font-size: 0.78rem;
    font-weight: 600;
    cursor: pointer;
    text-align: center;
    transition: all 0.15s;
}
.lang-btn.active {
    background: rgba(56,139,253,0.15);
    border-color: #388bfd;
    color: #388bfd;
}

/* ── Cards ── */
.qc-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.qc-card-title {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8b949e;
    margin-bottom: 8px;
}

/* ── Status badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.status-live {
    background: rgba(35,197,94,0.15);
    border: 1px solid #23c55e;
    color: #23c55e;
}
.status-stopped {
    background: rgba(139,148,158,0.1);
    border: 1px solid #30363d;
    color: #8b949e;
}
.pulse {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #23c55e;
    animation: pulse 1.4s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.7); }
}
.dot-stopped { width: 8px; height: 8px; border-radius: 50%; background: #8b949e; }

/* ── Big metric ── */
.big-metric { text-align: center; padding: 8px 0; }
.big-metric .value {
    font-size: 2rem; font-weight: 700;
    line-height: 1; color: #e6edf3;
}
.big-metric .label {
    font-size: 0.68rem; color: #8b949e;
    text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px;
}
.big-metric.alert .value { color: #f85149; }
.big-metric.ok    .value { color: #23c55e; }

/* ── Defect type pill ── */
.defect-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 2px 3px;
}
.pill-1 { background:rgba(248,81,73,0.15);   border:1px solid #f85149; color:#f85149; }
.pill-2 { background:rgba(255,166,0,0.15);   border:1px solid #ffa600; color:#ffa600; }
.pill-3 { background:rgba(255,212,0,0.15);   border:1px solid #ffd400; color:#ffd400; }
.pill-4 { background:rgba(188,123,255,0.15); border:1px solid #bc7bff; color:#bc7bff; }
.pill-5 { background:rgba(56,139,253,0.15);  border:1px solid #388bfd; color:#388bfd; }
.pill-ok{ background:rgba(35,197,94,0.1);    border:1px solid #23c55e; color:#23c55e; }

/* ── Alert feed row ── */
.alert-row {
    display: flex; align-items: center; gap: 10px;
    padding: 7px 0; border-bottom: 1px solid #21262d;
    font-size: 0.8rem;
}
.alert-time  { color:#8b949e; min-width:62px; font-size:0.7rem; font-variant-numeric:tabular-nums; }
.alert-conf  { color:#8b949e; margin-left:auto; font-variant-numeric:tabular-nums; }

/* ── Video frame wrapper ── */
.video-wrapper {
    background: #010409;
    border: 1px solid #30363d;
    border-radius: 8px;
    overflow: hidden;
    line-height: 0;
    min-height: 200px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.video-wrapper img { width: 100%; display: block; }
.video-placeholder-text {
    color: #30363d; font-size: 0.9rem;
    text-align: center; padding: 60px 20px;
    font-family: monospace; line-height: 1.6;
}

/* ── Streamlit overrides ── */
[data-testid="stButton"] button {
    border-radius: 6px; font-weight: 600;
    font-size: 0.82rem; letter-spacing: 0.04em; transition: all 0.15s;
}
[data-testid="stMetric"]   { background: transparent !important; }
hr                         { border-color: #21262d; }
[data-testid="stDataFrame"]{ border: 1px solid #30363d; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Language — must resolve before detector and layout
# ---------------------------------------------------------------------------

if "lang" not in st.session_state:
    st.session_state.lang = "en"


def t(key: str) -> str:
    """Return translated string for the current language."""
    return TRANSLATIONS.get(key, {}).get(st.session_state.lang, key)


# Language switcher rendered at the very top of the sidebar
with st.sidebar:
    st.markdown(
        '<style>'
        '#lang-switcher { display:flex; gap:6px; margin-bottom:4px; }'
        '#lang-switcher [data-testid="stButton"] { flex:1; }'
        '#lang-switcher [data-testid="stButton"] button {'
        '  font-size:0.75rem !important; padding:4px 0 !important;'
        '  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'
        '  letter-spacing:0.03em !important;'
        '}'
        '</style>'
        '<div id="lang-switcher">',
        unsafe_allow_html=True,
    )
    lang_col1, lang_col2 = st.columns(2)
    with lang_col1:
        if st.button("🇬🇧 EN", use_container_width=True,
                     type="primary" if st.session_state.lang == "en" else "secondary"):
            st.session_state.lang = "en"
            st.rerun()
    with lang_col2:
        if st.button("🇹🇭 ไทย", use_container_width=True,
                     type="primary" if st.session_state.lang == "th" else "secondary"):
            st.session_state.lang = "th"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Defect type label map — language-aware
_TYPE_LABEL: dict[int, str] = {
    1: t("defect_1"), 2: t("defect_2"), 3: t("defect_3"),
    4: t("defect_4"), 5: t("defect_5"),
}
_PILL_CLASS: dict[int, str] = {
    1: "pill-1", 2: "pill-2", 3: "pill-3", 4: "pill-4", 5: "pill-5",
}

# ---------------------------------------------------------------------------
# Detector (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_detector() -> DefectDetector:
    return DefectDetector()

detector = get_detector()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame_to_html(frame_rgb) -> str:
    """Encode RGB ndarray as base64 JPEG <img> — bypasses Streamlit media store."""
    _, buf = cv2.imencode(
        ".jpg",
        cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 82],
    )
    b64 = base64.b64encode(buf.tobytes()).decode()
    return f'<img src="data:image/jpeg;base64,{b64}" style="width:100%;display:block;">'


def _pill(type_id: int, label: str) -> str:
    cls = _PILL_CLASS.get(type_id, "pill-ok")
    return f'<span class="defect-pill {cls}">{label}</span>'


def _status_badge(live: bool) -> str:
    if live:
        return (
            f'<span class="status-badge status-live">'
            f'<span class="pulse"></span>{t("status_live")}</span>'
        )
    return (
        f'<span class="status-badge status-stopped">'
        f'<span class="dot-stopped"></span>{t("status_stopped")}</span>'
    )


def _disk_image_to_html(path: str, caption: str = "") -> str:
    """Load a JPEG/PNG from disk and return a base64 HTML img block.
    Bypasses Streamlit's media store — safe to call during rapid reruns."""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    cap_html = (
        f'<div style="font-size:0.65rem;color:#8b949e;padding:4px 0;'
        f'text-align:center;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">'
        f"{caption}</div>"
        if caption else ""
    )
    return (
        f'<div style="background:#010409;border:1px solid #30363d;border-radius:6px;'
        f'overflow:hidden;margin-bottom:8px;">'
        f'<img src="data:image/{mime};base64,{b64}" style="width:100%;display:block;">'
        f"{cap_html}</div>"
    )


def _big_metric(value, label, style="") -> str:
    cls = f"big-metric {style}".strip()
    return (
        f'<div class="{cls}">'
        f'<div class="value">{value}</div>'
        f'<div class="label">{label}</div>'
        f"</div>"
    )

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

_defaults: dict = {
    "processing":      False,
    "video_source":    None,
    "frame_queue":     None,
    "stop_event":      None,
    "capture_thread":  None,
    "frame_count":     0,
    "defect_count":    0,
    "latest_frame":    None,
    "latest_defects":  [],
    "alert_feed":      [],
    "type_counts":     defaultdict(int),
    "save_images":     False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Capture worker
# ---------------------------------------------------------------------------

def _resize_frame(frame, display_height: int):
    """Resize frame to fit within a 16:9 bounding box of the given height."""
    max_h = display_height
    max_w = display_height * 16 // 9
    h, w  = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    return cv2.resize(frame, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)


def _process_and_enqueue(frame, frame_count, frame_queue, detector, display_height):
    """Resize, detect, convert to RGB, and put onto queue (non-blocking)."""
    frame_small = _resize_frame(frame, display_height)
    marked, defect_infos = detector.process_frame(frame_small)
    frame_rgb = cv2.cvtColor(marked, cv2.COLOR_BGR2RGB)
    try:
        frame_queue.put_nowait((frame_rgb, defect_infos, frame_count))
    except queue_module.Full:
        pass


def _capture_worker(
    video_source,
    frame_queue: queue_module.Queue,
    stop_event: threading.Event,
    detector: DefectDetector,
    target_fps: int = 15,
    display_height: int = 480,
    source_type: str = "cv2",   # "cv2" | "images" | "gige"
    images_fps: int = 10,
) -> None:
    # ── Image sequence ───────────────────────────────────────────────────────
    if source_type == "images":
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        files = []
        for ext in exts:
            files += glob_module.glob(os.path.join(video_source, ext))
            files += glob_module.glob(os.path.join(video_source, ext.upper()))
        files = sorted(set(files))
        if not files:
            frame_queue.put("error:no_source")
            return
        interval = 1.0 / max(1, images_fps)
        frame_count = 0
        for fpath in files:
            if stop_event.is_set():
                break
            frame = cv2.imread(fpath)
            if frame is None:
                continue
            frame_count += 1
            _process_and_enqueue(frame, frame_count, frame_queue, detector, display_height)
            time.sleep(interval)
        frame_queue.put(None)
        return

    # ── GigE Vision via harvesters (GenICam / GigE Vision standard) ─────────
    if source_type == "gige":
        try:
            import numpy as np
            from harvesters.core import Harvester
        except ImportError:
            frame_queue.put("error:no_source")
            return
        cti_path, camera_index = video_source   # tuple passed from sidebar
        h = Harvester()
        try:
            h.add_file(cti_path)
            h.update()
            if not h.device_info_list:
                frame_queue.put("error:no_source")
                return
            idx = min(camera_index, len(h.device_info_list) - 1)
            ia = h.create(idx)
            ia.start()
            frame_count = 0
            try:
                while not stop_event.is_set():
                    with ia.fetch(timeout=2.0) as buffer:
                        component = buffer.payload.components[0]
                        data = component.data
                        h_px, w_px = component.height, component.width
                        channels = data.size // (h_px * w_px)
                        frame = data.reshape(h_px, w_px, channels) if channels > 1 \
                                else cv2.cvtColor(
                                    data.reshape(h_px, w_px), cv2.COLOR_GRAY2BGR
                                )
                        if channels == 3:
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        frame_count += 1
                        _process_and_enqueue(
                            frame, frame_count, frame_queue, detector, display_height
                        )
            finally:
                ia.stop()
                ia.destroy()
        finally:
            h.reset()
        frame_queue.put(None)
        return

    # ── OpenCV-compatible sources: webcam / video file / RTSP / MJPEG ───────
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        frame_queue.put("error:no_source")
        return

    source_fps  = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_skip  = max(1, int(source_fps / target_fps))
    frame_count = 0

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
            _process_and_enqueue(frame, frame_count, frame_queue, detector, display_height)
    finally:
        cap.release()
        frame_queue.put(None)


def _start_capture(
    video_source,
    display_height: int = 480,
    source_type: str = "cv2",
    images_fps: int = 10,
) -> None:
    _stop_capture()
    stop_event  = threading.Event()
    frame_queue = queue_module.Queue(maxsize=4)
    thread = threading.Thread(
        target=_capture_worker,
        args=(video_source, frame_queue, stop_event, detector),
        kwargs={"display_height": display_height,
                "source_type": source_type,
                "images_fps": images_fps},
        daemon=True,
    )
    thread.start()
    st.session_state.update({
        "stop_event":     stop_event,
        "frame_queue":    frame_queue,
        "capture_thread": thread,
        "frame_count":    0,
        "defect_count":   0,
        "latest_frame":   None,
        "latest_defects": [],
        "alert_feed":     [],
        "type_counts":    defaultdict(int),
        "processing":     True,
    })


def _stop_capture() -> None:
    if st.session_state.stop_event:
        st.session_state.stop_event.set()
    thread = st.session_state.get("capture_thread")
    if thread and thread.is_alive():
        thread.join(timeout=2)
    st.session_state.update({
        "processing":     False,
        "capture_thread": None,
        "stop_event":     None,
        "frame_queue":    None,
    })

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("---")
    st.markdown(f"#### {t('section_source')}")

    source_options = [
        t("source_webcam"), t("source_file"), t("source_ip"),
        t("source_mjpeg"), t("source_images"), t("source_gige"),
    ]
    video_source_option = st.radio("source", source_options, label_visibility="collapsed")

    video_source_input = None
    uploaded_file      = None
    _source_type       = "cv2"   # passed to _start_capture
    _images_fps        = 10

    if video_source_option == t("source_webcam"):
        st.caption(t("local_only_note"))
        webcam_index = st.number_input(t("webcam_index"), min_value=0, value=0, step=1)
        video_source_input = int(webcam_index)

    elif video_source_option == t("source_file"):
        uploaded_file = st.file_uploader(t("upload_label"), type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file:
            temp_dir = "temp_videos"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            video_source_input = temp_path

    elif video_source_option == t("source_ip"):
        st.caption(t("local_only_note"))
        ip_url = st.text_input(t("rtsp_label"), placeholder=t("rtsp_placeholder"))
        video_source_input = ip_url or None

    elif video_source_option == t("source_mjpeg"):
        st.caption(t("local_only_note"))
        mjpeg_url = st.text_input(t("mjpeg_label"), placeholder=t("mjpeg_placeholder"))
        video_source_input = mjpeg_url or None
        # OpenCV reads MJPEG streams the same way as RTSP — source_type stays "cv2"

    elif video_source_option == t("source_images"):
        _source_type = "images"
        folder_path = st.text_input(
            t("images_label"),
            placeholder=t("images_placeholder"),
            help=t("images_help"),
        )
        _images_fps = st.slider(t("images_fps"), min_value=1, max_value=30, value=10)
        video_source_input = folder_path or None

    elif video_source_option == t("source_gige"):
        _source_type = "gige"
        st.caption(t("local_only_note"))
        cti_path = st.text_input(
            t("gige_cti_label"),
            placeholder=t("gige_cti_placeholder"),
            help=t("gige_cti_help"),
        )
        gige_index = st.number_input(
            t("gige_index"), min_value=0, value=0, step=1,
            help=t("gige_index_help"),
        )
        # Pass as tuple; worker unpacks (cti_path, camera_index)
        video_source_input = (cti_path, int(gige_index)) if cti_path else None
        try:
            from harvesters.core import Harvester  # noqa: F401
        except ImportError:
            st.warning(t("gige_help") + "  \n`pip install harvesters`")

    st.markdown("---")
    st.markdown(f"#### {t('section_detection')}")

    detector.CONFIDENCE_THRESHOLD = st.slider(
        t("conf_threshold"), 0.3, 1.0,
        float(detector.CONFIDENCE_THRESHOLD), 0.05,
        help=t("conf_help"),
    )
    detector.DARK_THRESHOLD = st.slider(
        t("dark_threshold"), 10, 120,
        int(detector.DARK_THRESHOLD), 5,
        help=t("dark_help"),
    )
    detector.BRIGHT_THRESHOLD = st.slider(
        t("bright_threshold"), 150, 255,
        int(detector.BRIGHT_THRESHOLD), 5,
    )
    with st.expander(t("area_filters")):
        detector.MIN_DEFECT_AREA = st.slider(
            t("min_area"), 5, 200, int(detector.MIN_DEFECT_AREA), 5,
        )
        detector.MAX_DEFECT_AREA = st.slider(
            t("max_area"), 500, 20000, int(detector.MAX_DEFECT_AREA), 500,
        )

    st.markdown("---")
    st.markdown(f"#### {t('section_display')}")
    display_height = st.select_slider(
        t("feed_height"), options=[240, 360, 480, 720], value=480,
    )
    save_images = st.toggle(
        t("save_images_toggle"),
        value=st.session_state.save_images,
        help=t("save_images_help"),
    )
    st.session_state.save_images = save_images

# Sync toggle → detector (runs after sidebar, so value is current on every rerun)
detector.save_images = st.session_state.save_images

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

hcol1, hcol2 = st.columns([3, 1])
with hcol1:
    st.markdown("# QCVision")
    st.markdown(t("app_subtitle"))
with hcol2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(_status_badge(st.session_state.processing), unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Control buttons
# ---------------------------------------------------------------------------

ctrl1, ctrl2, ctrl3, _ = st.columns([2, 2, 2, 6])

with ctrl1:
    start_btn = st.button(
        t("btn_start"), use_container_width=True,
        disabled=st.session_state.processing, type="primary",
    )
with ctrl2:
    stop_btn = st.button(
        t("btn_stop"), use_container_width=True,
        disabled=not st.session_state.processing,
    )
with ctrl3:
    clear_btn = st.button(t("btn_clear"), use_container_width=True)

if clear_btn:
    if os.path.exists(detector.log_file):
        os.remove(detector.log_file)
        detector._initialize_logging()

if stop_btn:
    _stop_capture()

if start_btn:
    if video_source_input is None or video_source_input == "":
        st.warning(t("no_source_warning"))
    else:
        st.session_state.video_source = video_source_input
        _start_capture(
            video_source_input,
            display_height=display_height,
            source_type=_source_type,
            images_fps=_images_fps,
        )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

left, right = st.columns([3, 2], gap="medium")

with left:
    feed_placeholder   = st.empty()
    status_placeholder = st.empty()

# Idle placeholder
if not st.session_state.processing and st.session_state.latest_frame is None:
    with feed_placeholder:
        st.markdown(
            f'<div class="video-wrapper">'
            f'<div class="video-placeholder-text">● {t("feed_idle_title")}<br>'
            f'<small>{t("feed_idle_sub")}</small></div></div>',
            unsafe_allow_html=True,
        )

with right:
    metrics_placeholder   = st.empty()
    breakdown_placeholder = st.empty()
    alertfeed_placeholder = st.empty()


def _render_stats() -> None:
    fc    = st.session_state.frame_count
    dc    = st.session_state.defect_count
    rate  = dc / max(1, fc) * 100
    style = "alert" if rate > 10 else "ok"

    with metrics_placeholder.container():
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(_big_metric(fc, t("metric_frames")), unsafe_allow_html=True)
        with m2:
            st.markdown(_big_metric(dc, t("metric_events"), style), unsafe_allow_html=True)
        with m3:
            st.markdown(_big_metric(f"{rate:.1f}%", t("metric_rate"), style), unsafe_allow_html=True)

    counts = st.session_state.type_counts
    if counts:
        rows = "".join(
            f'<div class="alert-row">'
            f'{_pill(tid, _TYPE_LABEL.get(tid, str(tid)))}'
            f'<span class="alert-conf">{cnt}</span>'
            f"</div>"
            for tid, cnt in sorted(counts.items())
        )
        with breakdown_placeholder.container():
            st.markdown(
                f'<div class="qc-card">'
                f'<div class="qc-card-title">{t("card_by_type")}</div>'
                f"{rows}</div>",
                unsafe_allow_html=True,
            )

    feed = st.session_state.alert_feed[-8:][::-1]
    if feed:
        rows = "".join(
            f'<div class="alert-row">'
            f'<span class="alert-time">{a["time"]}</span>'
            f'{_pill(a["type_id"], a["type_name"])}'
            f'<span class="alert-conf">{a["confidence"]:.2f}</span>'
            f"</div>"
            for a in feed
        )
        with alertfeed_placeholder.container():
            st.markdown(
                f'<div class="qc-card">'
                f'<div class="qc-card-title">{t("card_recent")}</div>'
                f"{rows}</div>",
                unsafe_allow_html=True,
            )


_render_stats()

# ---------------------------------------------------------------------------
# Live feed loop — one frame per rerun
# ---------------------------------------------------------------------------

if st.session_state.processing:
    fq = st.session_state.frame_queue

    if fq is not None:
        try:
            item = fq.get(timeout=0.12)
        except queue_module.Empty:
            item = "no_frame"

        if item is None:
            _stop_capture()
            with status_placeholder:
                st.info(t("feed_ended"))
        elif item == "error:no_source":
            _stop_capture()
            with status_placeholder:
                st.error(t("feed_no_source"))
        elif item != "no_frame":
            frame_rgb, defect_infos, frame_count = item
            st.session_state.frame_count    = frame_count
            st.session_state.latest_frame   = frame_rgb
            st.session_state.latest_defects = defect_infos

            if defect_infos:
                st.session_state.defect_count += 1
                for d in defect_infos:
                    tid = next(
                        (k for k, v in detector.DEFECT_TYPES.items() if v == d["type"]),
                        0,
                    )
                    st.session_state.type_counts[tid] += 1
                    st.session_state.alert_feed.append({
                        "time":       datetime.now().strftime("%H:%M:%S"),
                        "type_id":    tid,
                        "type_name":  _TYPE_LABEL.get(tid, d["type"]),
                        "confidence": d["confidence"],
                    })

    if st.session_state.latest_frame is not None:
        with feed_placeholder:
            st.markdown(
                f'<div class="video-wrapper">{_frame_to_html(st.session_state.latest_frame)}</div>',
                unsafe_allow_html=True,
            )
        with status_placeholder:
            if st.session_state.latest_defects:
                names = "  ".join(
                    _pill(
                        next((k for k, v in detector.DEFECT_TYPES.items()
                              if v == d["type"]), 0),
                        d["type"],
                    )
                    for d in st.session_state.latest_defects
                )
                st.markdown(names, unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<span class="defect-pill pill-ok">{t("defect_clear")}</span>',
                    unsafe_allow_html=True,
                )

    _render_stats()

    if st.session_state.processing:
        st.rerun()

elif st.session_state.latest_frame is not None:
    with feed_placeholder:
        st.markdown(
            f'<div class="video-wrapper">{_frame_to_html(st.session_state.latest_frame)}</div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Defect log — collapsible
# ---------------------------------------------------------------------------

st.markdown("---")
with st.expander(t("log_expander"), expanded=False):
    if os.path.exists(detector.log_file):
        try:
            df = pd.read_csv(detector.log_file)
            if not df.empty:
                lc1, lc2, lc3 = st.columns(3)
                with lc1:
                    st.metric(t("log_total"), len(df))
                with lc2:
                    avg = pd.to_numeric(df["Confidence"], errors="coerce").mean()
                    st.metric(t("log_avg_conf"), f"{avg:.2f}")
                with lc3:
                    st.metric(t("log_types"), df["Defect_Type"].nunique())
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.download_button(
                    t("log_download"),
                    data=df.to_csv(index=False),
                    file_name=f"defect_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
            else:
                st.info(t("log_empty"))
        except Exception as e:
            st.error(f"{t('log_error')} {e}")
    else:
        st.info(t("log_no_file"))

# ---------------------------------------------------------------------------
# Defective images gallery
# Images are loaded from disk and encoded as base64 HTML to avoid
# Streamlit's in-memory media store (which causes MediaFileStorageError
# when files are evicted between rapid reruns).
# ---------------------------------------------------------------------------

with st.expander(t("gallery_expander"), expanded=False):
    image_dir = detector.image_dir
    if os.path.exists(image_dir):
        image_files = sorted(
            [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png"))],
            key=lambda x: os.path.getmtime(os.path.join(image_dir, x)),
            reverse=True,
        )
        if image_files:
            st.metric(t("gallery_count"), len(image_files))
            # Show most recent 12 images in a 3-column grid
            cols = st.columns(3)
            for idx, fname in enumerate(image_files[:12]):
                fpath = os.path.join(image_dir, fname)
                try:
                    html = _disk_image_to_html(fpath, caption=fname)
                    with cols[idx % 3]:
                        st.markdown(html, unsafe_allow_html=True)
                except Exception:
                    pass  # Skip unreadable files silently
        else:
            st.info(t("gallery_empty"))
    else:
        st.info(t("gallery_empty"))
