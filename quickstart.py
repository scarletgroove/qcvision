#!/usr/bin/env python3
"""
Quick Start Script for Paper Roll Defect Detection
Helps setup and calibrate the detection system
"""

import os
import sys
import argparse
import subprocess

def print_banner():
    """Print welcome banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║     Paper Roll Defect Detection - Quick Start Tool          ║
    ║     Real-time Inspection without ML Training                ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """Check if all required packages are installed"""
    print("\n📦 Checking dependencies...")
    
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'streamlit': 'streamlit',
        'pymodbus': 'pymodbus'
    }
    
    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        print("✅ Dependencies installed!")
    else:
        print("\n✅ All dependencies satisfied!")
    
    return len(missing) == 0

def setup_directories():
    """Create necessary directories"""
    print("\n📁 Setting up directories...")
    
    dirs = {
        'defective_images': 'Defect image storage',
        'temp_videos': 'Temporary video files'
    }
    
    for dir_path, description in dirs.items():
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"  ✅ Created: {dir_path} ({description})")
        else:
            print(f"  ✓ Already exists: {dir_path}")
    
    print("✅ Directory setup complete!")

def show_usage_options():
    """Show available usage options"""
    print("\n" + "="*60)
    print("📚 USAGE OPTIONS")
    print("="*60)
    
    options = {
        "1": {
            "name": "Quick Test (Webcam)",
            "command": "python -m src.realtime_defect_detector",
            "description": "Test detection on webcam with default settings"
        },
        "2": {
            "name": "Web Interface (Streamlit)",
            "command": "streamlit run streamlit_app.py",
            "description": "Graphical web interface with parameter tuning"
        },
        "3": {
            "name": "Calibration Mode",
            "command": "python calibrate_detector.py",
            "description": "Interactive parameter tuning wizard"
        },
        "4": {
            "name": "View Defect Log",
            "command": "python scripts/view_log.py",
            "description": "Analyze collected defect data"
        },
        "5": {
            "name": "Generate Report",
            "command": "python scripts/generate_report.py",
            "description": "Create quality inspection report"
        }
    }
    
    for key, option in options.items():
        print(f"\n  [{key}] {option['name']}")
        print(f"      📝 {option['description']}")
        print(f"      💻 {option['command']}")
    
    return options

def main():
    """Main entry point"""
    print_banner()
    
    parser = argparse.ArgumentParser(description='Defect Detection Quick Start')
    parser.add_argument('--test', action='store_true', help='Run quick test')
    parser.add_argument('--web', action='store_true', help='Start web interface')
    parser.add_argument('--calibrate', action='store_true', help='Run calibration')
    parser.add_argument('--check', action='store_true', help='Check only')
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    if args.check:
        print("✅ System ready!")
        return
    
    options = show_usage_options()
    
    if args.test:
        choice = "1"
    elif args.web:
        choice = "2"
    elif args.calibrate:
        choice = "3"
    else:
        print("\n" + "="*60)
        print("Select an option (1-5):")
        choice = input("Your choice: ").strip()
    
    if choice in options:
        option = options[choice]
        print(f"\n▶️  Starting: {option['name']}")
        print(f"Command: {option['command']}")
        print("\n" + "="*60)
        
        try:
            subprocess.run(option['command'], shell=True, check=False)
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopped by user")
    else:
        print("❌ Invalid choice")
        sys.exit(1)

if __name__ == "__main__":
    main()
