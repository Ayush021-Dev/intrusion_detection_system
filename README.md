# Intrusion Detection System

A real-time intrusion detection system using YOLOv8 and Streamlit. This system monitors a specified zone using a camera feed and detects when people enter the restricted area.

## Features

- Real-time object detection using YOLOv8
- Customizable detection zones
- Automatic screenshot capture of intrusions
- MySQL database logging
- Streamlit web interface
- Live camera feed monitoring

## Prerequisites

- Python 3.10
- MySQL Server
- Webcam

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Ayush021-Dev/intrusion_detection_system.git
cd intrusion_detection_system
```

2. Create and activate a virtual environment:
```bash
conda create -n intrusion_detection python=3.10
conda activate intrusion_detection
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up MySQL database:
```bash
mysql -u root -p < database/setup.sql
```

5. Update database configuration in `config.py`:
```python
DB_CONFIG = {
    'host': 'localhost',
    'user': 'your_username',
    'password': 'your_password',
    'database': 'intrusion_detection'
}
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Use the sidebar to:
   - View live camera feed
   - Adjust detection zone
   - View intrusion logs

## Project Structure

```
intrusion_detection_system/
│
├── app.py                   # Main Streamlit application
├── detection/
│   ├── __init__.py
│   ├── detector.py          # YOLOv8 implementation
│   └── zone_detector.py     # Zone detection logic
├── database/
│   ├── __init__.py
│   └── db_handler.py        # MySQL database operations
├── utils/
│   ├── __init__.py
│   └── image_utils.py       # Image processing utilities
├── config.py                # Configuration settings
├── requirements.txt         # Dependencies
└── README.md               # Project documentation
```

## Camera Configuration

All cameras (webcam or RTSP) are now configured via a `cameras.json` file in the project root. Example format:

```
[
  {
    "name": "Webcam",
    "camera_index": 0,
    "is_active": true,
    "zone_points": [[100, 100], [400, 100], [400, 300], [100, 300]]
  },
  {
    "name": "Office RTSP",
    "camera_index": "rtsp://user:pass@192.168.1.10:554/stream1",
    "is_active": true,
    "zone_points": [[50, 50], [300, 50], [300, 200], [50, 200]]
  }
]
```

- `camera_index` can be an integer (for local webcams) or a string (for RTSP URLs).
- The number of cameras is determined by the number of entries in this file.
- On startup, the app will sync the database with this file.

