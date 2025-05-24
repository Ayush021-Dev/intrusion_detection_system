-- Create the database
CREATE DATABASE IF NOT EXISTS intrusion_detection;

-- Use the database
USE intrusion_detection;

-- Create the intrusion_logs table
CREATE TABLE IF NOT EXISTS intrusion_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    camera_id VARCHAR(50) NOT NULL,
    image_path VARCHAR(255) NOT NULL
); 