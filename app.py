from flask import Flask, render_template, Response, jsonify, request, send_file
import cv2
import numpy as np
import time
from datetime import datetime, date
import os
import pandas as pd
import threading
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import DateTime
import json
import io
from pytz import timezone, utc

# Import from project modules
from detection.detector import ObjectDetector
from detection.zone_detector import ZoneDetector
from utils.image_utils import save_screenshot, add_timestamp, resize_image
from config import (
    DB_CONFIG, CAMERA_ID, CAMERA_INDEX, DETECTION_COOLDOWN,
    DEFAULT_ZONE_POINTS, CONFIDENCE_THRESHOLD, SCREENSHOT_DIR
)

app = Flask(__name__)

# Configure SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///intrusion.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define database models
class IntrusionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(DateTime, nullable=False, default=datetime.utcnow)
    camera_id = db.Column(db.String(50), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    event_type = db.Column(db.String(50), default='Detection')

# Global variables
PREVIOUS_PERSON_COUNT = 0
LAST_DETECTION_TIME = 0
PREVIOUS_FRAME = None
MOTION_THRESHOLD = 30
MIN_MOTION_AREA = 2000
MAX_MOTION_AREA = 100000
MIN_MOTION_WIDTH = 50
MIN_MOTION_HEIGHT = 50

# Initialize zone points
zone_points = DEFAULT_ZONE_POINTS

# Initialize detector
detector = ObjectDetector(confidence_threshold=CONFIDENCE_THRESHOLD)
zone_detector = ZoneDetector(zone_points)

def detect_motion(frame, zone_detector):
    global PREVIOUS_FRAME
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    if PREVIOUS_FRAME is None:
        PREVIOUS_FRAME = gray
        return False, frame
    
    frame_delta = cv2.absdiff(PREVIOUS_FRAME, gray)
    thresh = cv2.threshold(frame_delta, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    motion_detected = False
    motion_frame = frame.copy()
    
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        if (area < MIN_MOTION_AREA or area > MAX_MOTION_AREA or 
            w < MIN_MOTION_WIDTH or h < MIN_MOTION_HEIGHT):
            continue
            
        if zone_detector.is_in_zone((x, y, x + w, y + h)):
            motion_detected = True
            cv2.rectangle(motion_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(motion_frame, "Motion", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    PREVIOUS_FRAME = gray
    return motion_detected, motion_frame

def process_frame(frame):
    global LAST_DETECTION_TIME, PREVIOUS_PERSON_COUNT
    
    processed_frame = frame.copy()
    detections = detector.detect(processed_frame)
    processed_frame = zone_detector.draw_zone(processed_frame)
    
    people_in_zone = 0
    for detection in detections:
        if zone_detector.is_in_zone(detection['bbox']):
            people_in_zone += 1
    
    processed_frame = detector.draw_detections(processed_frame, detections, zone_detector)
    motion_detected, motion_frame = detect_motion(frame, zone_detector)
    
    cv2.putText(processed_frame, f"People in zone: {people_in_zone}", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if motion_detected:
        cv2.putText(processed_frame, "Significant Motion Detected!", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    activity_detected = people_in_zone > 0 or motion_detected
    current_time = time.time()
    person_count_changed = people_in_zone != PREVIOUS_PERSON_COUNT
    
    if (person_count_changed or motion_detected) and current_time - LAST_DETECTION_TIME > DETECTION_COOLDOWN:
        if people_in_zone > PREVIOUS_PERSON_COUNT:
            event_type = "Entry"
        elif people_in_zone < PREVIOUS_PERSON_COUNT:
            event_type = "Exit"
        elif motion_detected:
            event_type = "Significant Motion"
        else:
            event_type = "Change"
        
        cv2.putText(processed_frame, f"Event: {event_type}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        timestamped_frame = add_timestamp(processed_frame)
        image_path = save_screenshot(timestamped_frame, SCREENSHOT_DIR, event_type)
        
        log = IntrusionLog(
            camera_id=CAMERA_ID,
            image_path=image_path,
            event_type=event_type
        )
        from flask import current_app
        with app.app_context():
            db.session.add(log)
            db.session.commit()
        
        LAST_DETECTION_TIME = current_time
    
    PREVIOUS_PERSON_COUNT = people_in_zone
    return processed_frame, activity_detected

def generate_frames():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            processed_frame, _ = process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_zone', methods=['POST'])
def update_zone():
    global zone_points, zone_detector
    data = request.get_json()
    points = data.get('points')
    if not points or len(points) != 4 or not all(isinstance(x, list) and len(x) == 2 and all(isinstance(i, (int, float)) for i in x) for x in points):
        return jsonify({'status': 'error', 'message': 'Invalid zone points'}), 400
    # Convert all values to int
    zone_points = [[int(x[0]), int(x[1])] for x in points]
    zone_detector = ZoneDetector(zone_points)
    return jsonify({'status': 'success'})

@app.template_filter('localtime')
def localtime_filter(value):
    if value is None:
        return ''
    local_tz = timezone('Asia/Kolkata')
    if value.tzinfo is None:
        value = utc.localize(value)
    return value.astimezone(local_tz).strftime('%Y-%m-%d %H:%M:%S')

@app.route('/logs')
def view_logs():
    category = request.args.get('category', 'all')
    date_filter = request.args.get('date')
    query = IntrusionLog.query
    if category != 'all':
        query = query.filter_by(event_type=category)
    if date_filter:
        query = query.filter(db.func.date(IntrusionLog.timestamp) == date_filter)
    logs = query.order_by(IntrusionLog.timestamp.desc()).all()
    for log in logs:
        log.image_path = os.path.basename(log.image_path)
    return render_template('logs.html', logs=logs)

@app.route('/screenshots/<path:filename>')
def serve_screenshot(filename):
    return send_file(os.path.join(SCREENSHOT_DIR, filename))

@app.route('/export_logs')
def export_logs():
    category = request.args.get('category', 'all')
    date_filter = request.args.get('date')
    query = IntrusionLog.query
    if category != 'all':
        query = query.filter_by(event_type=category)
    if date_filter:
        query = query.filter(db.func.date(IntrusionLog.timestamp) == date_filter)
    logs = query.order_by(IntrusionLog.timestamp.desc()).all()
    data = []
    for log in logs:
        local_tz = timezone('Asia/Kolkata')
        ts = log.timestamp
        if ts.tzinfo is None:
            ts = utc.localize(ts)
        local_ts = ts.astimezone(local_tz).strftime('%Y-%m-%d %H:%M:%S')
        data.append({
            'ID': log.id,
            'Timestamp': local_ts,
            'Camera ID': log.camera_id,
            'Event Type': log.event_type,
            'Image Path': os.path.basename(log.image_path)
        })
    df = pd.DataFrame(data)
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return Response(
        output,
        mimetype='text/csv',
        headers={
            'Content-Disposition': 'attachment; filename=intrusion_logs.csv'
        }
    )

@app.template_filter('basename')
def basename_filter(path):
    return os.path.basename(path)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)