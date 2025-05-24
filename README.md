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
git clone <your-repo-url>
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

## License

This project is licensed under the MIT License - see the LICENSE file for details.
