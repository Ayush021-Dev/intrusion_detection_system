# Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'mysql',  # Change this to your MySQL password
    'database': 'intrusion_detection'
}

# Camera Configuration
CAMERA_ID = "camera_0"
CAMERA_INDEX = 0  # Default webcam

# Detection Configuration
DETECTION_COOLDOWN = 3  # seconds between consecutive detections
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for person detection

# Default zone points (rectangle in the middle)
DEFAULT_ZONE_POINTS = [(100, 100), (400, 100), (400, 300), (100, 300)]

# Screenshot directory
SCREENSHOT_DIR = "screenshots"