import mysql.connector
import os
from datetime import datetime
import streamlit as st

class DatabaseHandler:
    def __init__(self, host, user, password, database):
        try:
            # Create database if it doesn't exist
            self.init_connection = mysql.connector.connect(
                host=host,
                user=user,
                password=password
            )
            init_cursor = self.init_connection.cursor()
            init_cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
            init_cursor.close()
            self.init_connection.close()
            
            # Connect to the database
            self.connection = mysql.connector.connect(
                host=host,
                user=user,
                password=password,
                database=database
            )
            self.cursor = self.connection.cursor()
            self.init_tables()
        except Exception as e:
            st.error(f"Database connection error: {e}")
    
    def init_tables(self):
        """Initialize the database tables if they don't exist"""
        try:
            # Create intrusion_logs table
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS intrusion_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME NOT NULL,
                camera_id VARCHAR(50) NOT NULL,
                image_path VARCHAR(255) NOT NULL,
                event_type VARCHAR(50) DEFAULT 'Detection'
            )
            """)
            self.connection.commit()
        except Exception as e:
            st.error(f"Database initialization error: {e}")
    
    def log_intrusion(self, camera_id, image_path, event_type="Detection"):
        """Log an intrusion event to the database"""
        try:
            query = """
            INSERT INTO intrusion_logs (timestamp, camera_id, image_path, event_type)
            VALUES (%s, %s, %s, %s)
            """
            timestamp = datetime.now()
            self.cursor.execute(query, (timestamp, camera_id, image_path, event_type))
            self.connection.commit()
            return self.cursor.lastrowid
        except Exception as e:
            st.error(f"Database logging error: {e}")
            return None
    
    def get_all_logs(self):
        """Retrieve all intrusion logs from the database"""
        try:
            query = "SELECT * FROM intrusion_logs ORDER BY timestamp DESC"
            self.cursor.execute(query)
            return self.cursor.fetchall()
        except Exception as e:
            st.error(f"Error retrieving logs: {e}")
            return []
    
    def get_logs_by_date(self, date):
        """Retrieve logs for a specific date"""
        try:
            query = "SELECT * FROM intrusion_logs WHERE DATE(timestamp) = %s ORDER BY timestamp DESC"
            self.cursor.execute(query, (date,))
            return self.cursor.fetchall()
        except Exception as e:
            st.error(f"Error retrieving logs by date: {e}")
            return []
    
    def get_logs_by_event_type(self, event_type):
        """Retrieve logs for a specific event type"""
        try:
            query = "SELECT * FROM intrusion_logs WHERE event_type = %s ORDER BY timestamp DESC"
            self.cursor.execute(query, (event_type,))
            return self.cursor.fetchall()
        except Exception as e:
            st.error(f"Error retrieving logs by event type: {e}")
            return []
    
    def delete_log(self, log_id):
        """Delete a specific log entry"""
        try:
            query = "DELETE FROM intrusion_logs WHERE id = %s"
            self.cursor.execute(query, (log_id,))
            self.connection.commit()
            return True
        except Exception as e:
            st.error(f"Error deleting log: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        try:
            self.cursor.close()
            self.connection.close()
        except Exception as e:
            st.error(f"Error closing database connection: {e}")