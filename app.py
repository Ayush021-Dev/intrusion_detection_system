import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime, date
import os
import pandas as pd
import threading

# Import from project modules
from database.db_handler import DatabaseHandler
from detection.detector import ObjectDetector
from detection.zone_detector import ZoneDetector
from utils.image_utils import save_screenshot, add_timestamp, resize_image
from config import (
    DB_CONFIG, CAMERA_ID, CAMERA_INDEX, DETECTION_COOLDOWN,
    DEFAULT_ZONE_POINTS, CONFIDENCE_THRESHOLD, SCREENSHOT_DIR
)

# Set page configuration
st.set_page_config(
    page_title="Intrusion Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global variables
PREVIOUS_PERSON_COUNT = 0
LAST_DETECTION_TIME = 0

# Initialize session state
if 'zone_points' not in st.session_state:
    st.session_state['zone_points'] = DEFAULT_ZONE_POINTS

# Cache the model to prevent reloading
@st.cache_resource
def load_model():
    return ObjectDetector(confidence_threshold=CONFIDENCE_THRESHOLD)

# Initialize database handler
@st.cache_resource
def get_db_handler():
    return DatabaseHandler(
        DB_CONFIG['host'],
        DB_CONFIG['user'],
        DB_CONFIG['password'],
        DB_CONFIG['database']
    )

# Process frame function
def process_frame(frame, detector, zone_detector):
    global LAST_DETECTION_TIME, PREVIOUS_PERSON_COUNT
    
    # Make a copy of the frame
    processed_frame = frame.copy()
    
    # Detect objects
    detections = detector.detect(processed_frame)
    
    # Draw zone
    processed_frame = zone_detector.draw_zone(processed_frame)
    
    # Count people in the zone
    people_in_zone = 0
    for detection in detections:
        if zone_detector.is_in_zone(detection['bbox']):
            people_in_zone += 1
    
    # Draw detections
    processed_frame = detector.draw_detections(processed_frame, detections, zone_detector)
    
    # Add count text to frame
    cv2.putText(processed_frame, f"People in zone: {people_in_zone}", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Check if the number of people in the zone has changed
    current_time = time.time()
    person_count_changed = people_in_zone != PREVIOUS_PERSON_COUNT
    
    # Take screenshot if count changed and cooldown period passed
    if person_count_changed and current_time - LAST_DETECTION_TIME > DETECTION_COOLDOWN:
        # Determine if it's an entry or exit event
        if people_in_zone > PREVIOUS_PERSON_COUNT:
            event_type = "Entry"
        elif people_in_zone < PREVIOUS_PERSON_COUNT:
            event_type = "Exit"
        else:
            event_type = "Change"  # This shouldn't happen, but just in case
        
        # Add event type to the frame
        cv2.putText(processed_frame, f"Event: {event_type}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add timestamp
        timestamped_frame = add_timestamp(processed_frame)
        
        # Save screenshot
        image_path = save_screenshot(timestamped_frame, SCREENSHOT_DIR, event_type)
        
        # Log the intrusion with the event type
        db_handler = get_db_handler()
        db_handler.log_intrusion(CAMERA_ID, image_path, event_type)
        
        # Update the last detection time
        LAST_DETECTION_TIME = current_time
    
    # Update the previous count for the next frame
    PREVIOUS_PERSON_COUNT = people_in_zone
    
    return processed_frame, people_in_zone > 0

# Page: Home / Camera Feed
def camera_feed_page():
    st.subheader("Live Camera Feed")
    
    # Instructions
    st.write("""
    ### Instructions:
    - The green quadrilateral shows the detection zone
    - When a person enters or exits the zone, a screenshot will be saved
    - Adjust the zone in the "Adjust Zone" page
    - View detection logs in the "View Logs" page
    """)
    
    # Initialize detector and zone detector
    detector = load_model()
    zone_detector = ZoneDetector(st.session_state['zone_points'])
    
    # Camera selection (currently only one camera)
    st.write("Camera: Default Webcam (ID: {})".format(CAMERA_ID))
    
    # Create placeholders for the video feed
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Create a stop button
    col1, col2 = st.columns([1, 4])
    with col1:
        stop_button = st.button("Stop Camera")
    
    # Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        st.error("Error: Could not open webcam")
        return
    
    # Main loop for camera feed
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break
        
        # Process the frame
        processed_frame, intrusion = process_frame(frame, detector, zone_detector)
        
        # Convert to RGB for display
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame
        frame_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
        
        # Update status
        status = "Status: 🔴 Person(s) Detected in Zone" if intrusion else "Status: 🟢 No Intrusion"
        status_placeholder.write(status)
        
        # Short sleep to reduce CPU usage
        time.sleep(0.03)
    
    # Release resources
    cap.release()
    st.write("Camera stopped")

# Page: Adjust Zone
def zone_adjustment_page():
    st.subheader("Adjust Detection Zone")
    
    # Instructions
    st.write("""
    ### Instructions:
    - Use the sliders to adjust the 4 corner points of the detection zone
    - The changes will be visible in the preview
    - Click "Save Zone Settings" to apply your changes
    """)
    
    # Get current zone points
    zone_points = st.session_state['zone_points']
    
    # Get frame dimensions from camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        st.error("Failed to capture video from camera")
        return
    
    height, width = frame.shape[:2]
    
        # Create placeholders for zone points
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Point 1")
        x1 = st.slider("Point 1 X", 0, width, zone_points[0][0], key="x1")
        y1 = st.slider("Point 1 Y", 0, height, zone_points[0][1], key="y1")
        
        st.write("Point 2")
        x2 = st.slider("Point 2 X", 0, width, zone_points[1][0], key="x2")
        y2 = st.slider("Point 2 Y", 0, height, zone_points[1][1], key="y2")
    
    with col2:
        st.write("Point 3")
        x3 = st.slider("Point 3 X", 0, width, zone_points[2][0], key="x3")
        y3 = st.slider("Point 3 Y", 0, height, zone_points[2][1], key="y3")
        
        st.write("Point 4")
        x4 = st.slider("Point 4 X", 0, width, zone_points[3][0], key="x4")
        y4 = st.slider("Point 4 Y", 0, height, zone_points[3][1], key="y4")
    
    # Update zone points
    new_zone_points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    
    # Display a preview
    cap = cv2.VideoCapture(CAMERA_INDEX)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Create a zone detector with the new points
        temp_zone_detector = ZoneDetector(new_zone_points)
        
        # Draw the zone on the frame
        preview_frame = temp_zone_detector.draw_zone(frame.copy())
        
        # Display the preview
        st.image(cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB), caption="Zone Preview", use_column_width=True)
    
    # Save button
    if st.button("Save Zone Settings"):
        st.session_state['zone_points'] = new_zone_points
        st.success("Zone settings saved!")
        
    # Reset button
    if st.button("Reset to Default"):
        st.session_state['zone_points'] = DEFAULT_ZONE_POINTS
        st.success("Zone reset to default!")
        st.experimental_rerun()

# Page: View Logs
def view_logs_page():
    st.subheader("Intrusion Logs")
    
    # Get database handler
    db_handler = get_db_handler()
    
    # Add filters
    col1, col2 = st.columns(2)
    
    with col1:
        # Date filter
        filter_date = st.date_input("Filter by Date", value=None)
    
    with col2:
        # Event type filter
        filter_type = st.selectbox(
            "Filter by Event Type",
            ["All", "Entry", "Exit"]
        )
    
    # Get logs based on filters
    if filter_date is not None and filter_type != "All":
        # Filter by both date and type
        # Convert datetime.date to string for MySQL query
        date_str = filter_date.strftime("%Y-%m-%d")
        logs = db_handler.get_logs_by_date(date_str)
        logs = [log for log in logs if log[4] == filter_type]  # Filter by event type
    elif filter_date is not None:
        # Filter by date only
        date_str = filter_date.strftime("%Y-%m-%d")
        logs = db_handler.get_logs_by_date(date_str)
    elif filter_type != "All":
        # Filter by event type only
        logs = db_handler.get_logs_by_event_type(filter_type)
    else:
        # No filters, get all logs
        logs = db_handler.get_all_logs()
    
    # Check if we have logs
    if not logs:
        st.info("No intrusion logs found matching the criteria.")
        return
    
    # Convert to DataFrame for better display
    df = pd.DataFrame(logs, columns=["ID", "Timestamp", "Camera ID", "Image Path", "Event Type"])
    
    # Display logs
    st.dataframe(df)
    
    # Show images
    st.subheader("Intrusion Screenshots")
    
    # Create columns for displaying images
    cols = st.columns(3)
    
    for i, (id, timestamp, camera_id, image_path, event_type) in enumerate(logs[:9]):  # Show up to 9 images
        try:
            img = cv2.imread(image_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Add caption with timestamp and event type
                caption = f"{timestamp} - {event_type}"
                
                # Display image
                cols[i % 3].image(img_rgb, caption=caption, use_column_width=True)
                
                # Add button to delete log
                if cols[i % 3].button(f"Delete Log #{id}", key=f"del_{id}"):
                    if db_handler.delete_log(id):
                        st.success(f"Log #{id} deleted successfully!")
                        st.experimental_rerun()
                    else:
                        st.error(f"Failed to delete log #{id}")
            else:
                cols[i % 3].error(f"Image not found: {image_path}")
        except Exception as e:
            cols[i % 3].error(f"Error loading image: {e}")
    
    # Export to CSV button
    if st.button("Export Logs to CSV"):
        csv_file = "intrusion_logs.csv"
        df.to_csv(csv_file, index=False)
        st.success(f"Logs exported to {csv_file}")

# Page: Settings
def settings_page():
    st.subheader("System Settings")
    
    # Detection settings
    st.write("### Detection Settings")
    
    # Confidence threshold
    confidence = st.slider(
        "Detection Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=CONFIDENCE_THRESHOLD,
        step=0.05,
        help="Minimum confidence level for person detection"
    )
    
    # Cooldown period
    cooldown = st.slider(
        "Detection Cooldown (seconds)",
        min_value=1,
        max_value=10,
        value=DETECTION_COOLDOWN,
        step=1,
        help="Minimum time between consecutive detections"
    )
    
    # Camera settings
    st.write("### Camera Settings")
    
    # Camera ID
    camera_id = st.text_input("Camera ID", value=CAMERA_ID)
    
    # Database settings
    st.write("### Database Settings")
    
    # Database host
    db_host = st.text_input("Database Host", value=DB_CONFIG['host'])
    
    # Database user
    db_user = st.text_input("Database User", value=DB_CONFIG['user'])
    
    # Database password
    db_password = st.text_input("Database Password", value=DB_CONFIG['password'], type="password")
    
    # Save settings button
    if st.button("Save Settings"):
        # In a real application, you would save these to a config file
        st.success("Settings saved! (Note: In a production app, these would be saved to a config file)")
        
        # For now, we'll just update the session state
        st.session_state['confidence'] = confidence
        st.session_state['cooldown'] = cooldown
        st.session_state['camera_id'] = camera_id
        st.session_state['db_config'] = {
            'host': db_host,
            'user': db_user,
            'password': db_password,
            'database': DB_CONFIG['database']
        }

# Main function
def main():
    # App title
    st.title("Intrusion Detection System")
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["Home", "Adjust Zone", "View Logs", "Settings"]
    )
    
    # Display system information in sidebar
    with st.sidebar.expander("System Information"):
        st.write("Camera ID:", CAMERA_ID)
        st.write("Model:", "YOLOv8n")
        st.write("Detection Cooldown:", f"{DETECTION_COOLDOWN} seconds")
        st.write("Confidence Threshold:", f"{CONFIDENCE_THRESHOLD}")
        st.write("Database:", DB_CONFIG['database'])
    
    # Display page based on selection
    if page == "Home":
        camera_feed_page()
    elif page == "Adjust Zone":
        zone_adjustment_page()
    elif page == "View Logs":
        view_logs_page()
    elif page == "Settings":
        settings_page()

# Run the application
if __name__ == "__main__":
    main()