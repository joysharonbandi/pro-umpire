
# app.py

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import numpy as np
from ultralytics import YOLO
import math
import time
import json
from collections import deque
import uuid

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load YOLO model
model_path = os.path.join('runs', 'detect', 'train9', 'weights', 'best.pt')
model = YOLO(model_path)  # Use a smaller model for faster processing
print("Model loaded successfully")

class FixedSizeQueue:
    def __init__(self, max_size):
        self.queue = deque(maxlen=max_size)
    
    def add(self, item):
        self.queue.append(item)
    
    def clear(self):
        self.queue.clear()

    def get_queue(self):
        return list(self.queue)
    
    def __len__(self):
        return len(self.queue)

def angle_between_lines(m1, m2=1):
    if m1 != -1/m2:
        angle = math.degrees(math.atan(abs((m2 - m1) / (1 + m1 * m2))))
        return angle
    else:
        return 90.0

def calculate_speed(p1, p2, fps, pixels_per_meter=100):
    """
    Calculate speed in meters per second
    p1, p2: points in (x,y) format
    fps: frames per second
    pixels_per_meter: conversion factor (needs calibration)
    """
    # Calculate distance in pixels
    distance_pixels = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    # Convert to meters
    distance_meters = distance_pixels / pixels_per_meter
    
    # Calculate time between frames
    time_seconds = 1 / fps
    
    # Calculate speed
    speed_mps = distance_meters / time_seconds
    
    # Convert to km/h
    speed_kmh = speed_mps * 3.6
    
    return speed_kmh

video_progress = {}

# Add a route to get progress for a specific video
@app.route('/progress/<video_id>', methods=['GET'])
def get_progress(video_id):
    if video_id in video_progress:
        return jsonify({
            "success": True,
            "progress": video_progress[video_id]
        })
    else:
        return jsonify({
            "success": False,
            "error": "Video ID not found"
        })

# Modify the process_video function to update progress
def process_video(video_path, output_json_path, pixels_per_meter=100, video_id=None):
    """Process video and save tracking data to JSON file"""
    try:
        # Initialize progress at 0%
        if video_id:
            video_progress[video_id] = 0
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"success": False, "error": "Failed to open video file"}

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Dictionary to store all tracking data
        tracking_data = {
            "video_info": {
                "fps": fps,
                "frame_count": frame_count,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "pixels_per_meter": pixels_per_meter
            },
            "bowledBowls":[],
            "frames": {},
            "balls": [],
        }
        
        frame_idx = 0
        centroid_history = FixedSizeQueue(10)
        no_detection_count = 0
        current_ball_centroids = []
        
        # Process every 3rd frame for speed
        frame_step = 3
        
        while frame_idx < frame_count:
            # Set position to specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame with YOLO
            results = model.track(frame, persist=True, classes=[0], conf=0.35, verbose=False)  # Only track people
            boxes = results[0].boxes
            
            # Data for this frame
            frame_data = {
                "centroids": [],
                "bounding_boxes": [],
                "future_positions": [],
                "angle": 0
            }
            
            # Check if we detected objects in this frame
            if len(boxes) > 0:
                # Get the box info
                box_data = boxes.xyxy.cpu().numpy()
                
                for box in box_data:
                    x1, y1, x2, y2 = box
                    
                    centroid_x = int((x1+x2)/2)
                    centroid_y = int((y1+y2)/2)
                    
                    centroid_history.add((centroid_x, centroid_y))
                    frame_data["centroids"].append([centroid_x, centroid_y])
                    frame_data["bounding_boxes"].append([int(x1), int(y1), int(x2), int(y2)])
                    
                    # Add to current ball centroids
                    current_ball_centroids.append([centroid_x, centroid_y, frame_idx])
                
                # Reset no detection counter since we found objects
                no_detection_count = 0
            else:
                # Increment no detection counter
                no_detection_count += 1
                
                # If we've had multiple frames with no detection and we have data for the current trajectory,
                # save this trajectory and prepare for the next one
                if no_detection_count > 15 and len(current_ball_centroids) > 0:
                    tracking_data["balls"].append(current_ball_centroids)
                    tracking_data["bowledBowls"].append(current_ball_centroids)
                    current_ball_centroids = []
                    centroid_history.clear()
            
            # Calculate angles and future positions
            angle = 0
            if len(centroid_history) > 1:
                centroid_list = centroid_history.get_queue()
                x_diff = centroid_list[-1][0] - centroid_list[-2][0]
                y_diff = centroid_list[-1][1] - centroid_list[-2][1]
                
                if x_diff != 0:
                    m1 = y_diff/x_diff
                    if m1 == 1:
                        angle = 90
                    elif m1 != 0:
                        angle = 90-angle_between_lines(m1)
                
                future_positions = [centroid_list[-1]]
                for i in range(1, 5):
                    future_pos = (
                        centroid_list[-1][0] + x_diff * i,
                        centroid_list[-1][1] + y_diff * i
                    )
                    future_positions.append(future_pos)
                    frame_data["future_positions"].append([int(future_pos[0]), int(future_pos[1])])
                
                frame_data["angle"] = angle
            
            # Store frame data
            tracking_data["frames"][str(frame_idx)] = frame_data
            
            # Update for next iteration - skip frames for speed
            frame_idx += frame_step
            
            # Calculate and update progress
            progress = (frame_idx / frame_count) * 100
            if video_id:
                video_progress[video_id] = progress
            
            # Print progress every 30 frames
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx}/{frame_count} frames ({progress:.1f}%)")
        
        # Add the last trajectory if it exists
        if len(current_ball_centroids) > 0:
            tracking_data["balls"].append(current_ball_centroids)
        
        # Calculate speeds for all trajectories
        for ball_idx, ball in enumerate(tracking_data["balls"]):
            # Add speed information to each trajectory point
            ball_with_speed = []
            for i in range(len(ball)):
                if i > 0:
                    # Current and previous positions
                    curr_pos = (ball[i][0], ball[i][1])  
                    prev_pos = (ball[i-1][0], ball[i-1][1])
                    
                    # Calculate time difference between frames
                    curr_frame = ball[i][2]
                    prev_frame = ball[i-1][2]
                    frame_diff = curr_frame - prev_frame
                    
                    # If frames are consecutive
                    if frame_diff > 0:
                        # Calculate speed considering potential frame gaps
                        effective_fps = fps / frame_diff
                        speed = calculate_speed(prev_pos, curr_pos, effective_fps, pixels_per_meter)
                    else:
                        speed = 0.0
                else:
                    speed = 0.0  # First position has no speed
                    
                # Store position with speed [x, y, frame_idx, speed]
                ball_with_speed.append([ball[i][0], ball[i][1], ball[i][2], speed])
                
            # Update the trajectory data with speed information
            tracking_data["balls"][ball_idx] = ball_with_speed
        
        # Save all tracking data to file
        with open(output_json_path, 'w') as f:
            json.dump(tracking_data, f)
        
        # Set progress to 100% when complete
        if video_id:
            video_progress[video_id] = 100
            
        cap.release()
        print(f"Processing complete. Data saved to {output_json_path}")
        return {"success": True, "file": os.path.basename(output_json_path)}
    
    except Exception as e:
        # Make sure to mark progress as failed
        if video_id:
            video_progress[video_id] = -1  # -1 can indicate an error
        
        print(f"Error processing video: {str(e)}")
        return {"success": False, "error": str(e)}

# Modify the upload route to pass the video_id to process_video
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"})
    
    pixels_per_meter = request.form.get('pixelsPerMeter', 100)
    try:
        pixels_per_meter = float(pixels_per_meter)
    except ValueError:
        pixels_per_meter = 100
    
    # Save uploaded file
    unique_id = str(uuid.uuid4())
    _, ext = os.path.splitext(file.filename)
    video_filename = f"{unique_id}{ext}"
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    file.save(video_path)
    
    # Initialize progress for this video
    video_progress[unique_id] = 0
    
    # Process video - now passing the video_id for progress tracking
    output_json = os.path.join(RESULTS_FOLDER, f"{unique_id}.json")
    
    # You might want to run this in a separate thread if it's a long-running process
    # For simplicity, I'm keeping it synchronous here
    result = process_video(video_path, output_json, pixels_per_meter, unique_id)
    
    if result["success"]:
        return jsonify({
            "success": True, 
            "videoId": unique_id,
            "message": "Video processed successfully"
        })
    else:
        return jsonify(result)
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"})
    
    pixels_per_meter = request.form.get('pixelsPerMeter', 100)
    try:
        pixels_per_meter = float(pixels_per_meter)
    except ValueError:
        pixels_per_meter = 100
    
    # Save uploaded file
    unique_id = str(uuid.uuid4())
    _, ext = os.path.splitext(file.filename)
    video_filename = f"{unique_id}{ext}"
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    file.save(video_path)
    
    # Process video
    output_json = os.path.join(RESULTS_FOLDER, f"{unique_id}.json")
    result = process_video(video_path, output_json, pixels_per_meter)
    
    if result["success"]:
        return jsonify({
            "success": True, 
            "videoId": unique_id,
            "message": "Video processed successfully"
        })
    else:
        return jsonify(result)

@app.route('/results/<file_id>', methods=['GET'])
def get_results(file_id):
    # Security check - make sure file_id doesn't contain path traversal
    if '..' in file_id or '/' in file_id:
        return jsonify({"success": False, "error": "Invalid file ID"})
    
    json_path = os.path.join(RESULTS_FOLDER, f"{file_id}.json")
    if not os.path.exists(json_path):
        return jsonify({"success": False, "error": "Results not found"})
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return jsonify({"success": True, "data": data})

@app.route('/video/<file_id>', methods=['GET'])
def get_video(file_id):
    # Find video file with this ID
    for filename in os.listdir(UPLOAD_FOLDER):
        if filename.startswith(file_id):
            return send_from_directory(UPLOAD_FOLDER, filename)
    
    return jsonify({"success": False, "error": "Video not found"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)