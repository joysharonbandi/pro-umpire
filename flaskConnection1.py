
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
import threading

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

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    import math
    if(point1 is not None and point2 is not None):
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    else:
        return None
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
video_status = {}  # To track the status (queued, processing, completed, error)

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
    
    # Initialize progress tracking
    video_progress[unique_id] = 0
    video_status[unique_id] = "queued"
    
    # Define the output path
    output_json = os.path.join(RESULTS_FOLDER, f"{unique_id}.json")
    
    # Start a background thread to process the video
    thread = threading.Thread(
        target=process_video_thread,
        args=(video_path, output_json, pixels_per_meter, unique_id)
    )
    thread.daemon = True  # Make the thread a daemon so it doesn't block program exit
    thread.start()
    
    # Return immediately with the video ID
    return jsonify({
        "success": True,
        "videoId": unique_id,
        "message": "Video upload successful. Processing started in background."
    })

# Function to run the processing in a background thread
def process_video_thread(video_path, output_json_path, pixels_per_meter, video_id):
    try:
        video_status[video_id] = "processing"
        result = process_video(video_path, output_json_path, pixels_per_meter, video_id)
        if result["success"]:
            video_status[video_id] = "completed"
        else:
            video_status[video_id] = "error"
            video_progress[video_id] = -1
    except Exception as e:
        print(f"Thread error processing video: {str(e)}")
        video_status[video_id] = "error"
        video_progress[video_id] = -1

# Add a route to get progress for a specific video
@app.route('/progress/<video_id>', methods=['GET'])
def get_progress(video_id):
    if video_id in video_progress:
        return jsonify({
            "success": True,
            "progress": video_progress[video_id],
            "status": video_status.get(video_id, "unknown")
        })
    else:
        return jsonify({
            "success": False,
            "error": "Video ID not found"
        })

# Keep the process_video function mostly the same, just update the progress reporting
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
            "bowlingMaxLength":[]
        }
        
        frame_idx = 0
        centroid_history = FixedSizeQueue(10)
        no_detection_count = 0
        current_ball_centroids = []
        distance_history = []
        max_distance_percent = 0
        max_distance_position = None
        
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



                    frame_height = frame.shape[0]
                    top_portion_height = frame_height * 0.70

                    if centroid_y <= top_portion_height:
                        distance_from_top = centroid_y
                        distance_percent = (distance_from_top / top_portion_height) * 100
                        distance_history.append(distance_percent)

                        if distance_percent > max_distance_percent:
                            max_distance_percent = distance_percent
                            max_distance_position = (centroid_x, centroid_y)


                    
                    # Add to current ball centroids
                    current_ball_centroids.append([centroid_x, centroid_y, frame_idx])
                    print(current_ball_centroids,"centroids123 ")
                
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
                    tracking_data["bowlingMaxLength"].append(calculate_distance((310,421),max_distance_position))
                    distance_history = []
                    max_distance_percent = 0
                    max_distance_position = None
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
        
        # Continue with the rest of the processing...
        # [Code for adding last trajectory and calculating speeds remains the same]
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

# Add an endpoint to retrieve the processing results
@app.route('/results/<video_id>', methods=['GET'])
def get_results(video_id):
    try:
        output_json = os.path.join(RESULTS_FOLDER, f"{video_id}.json")
        
        # Check if the file exists and if processing is complete
        if os.path.exists(output_json) and video_status.get(video_id) == "completed":
            with open(output_json, 'r') as f:
                data = json.load(f)
            return jsonify({"success": True, "data": data})
        elif video_status.get(video_id) == "error":
            return jsonify({"success": False, "error": "Processing failed"})
        else:
            return jsonify({"success": False, "error": "Processing not complete"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# Add an endpoint to serve the video files
@app.route('/video/<video_id>', methods=['GET'])
def get_video(video_id):
    # Find the video file for this ID
    for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
        video_path = os.path.join(UPLOAD_FOLDER, f"{video_id}{ext}")
        if os.path.exists(video_path):
            return send_from_directory(UPLOAD_FOLDER, f"{video_id}{ext}")
    
    return jsonify({"success": False, "error": "Video not found"})





# Add this new route to your Flask app
@app.route('/visualize/<video_id>', methods=['POST'])
def visualize_path(video_id):
    try:
        # Check if results exist
        output_json = os.path.join(RESULTS_FOLDER, f"{video_id}.json")
        if not os.path.exists(output_json):
            return jsonify({"success": False, "error": "Results not found"})
            
        # Load the tracking data
        with open(output_json, 'r') as f:
            tracking_data = json.load(f)
            
        # Get the reference pitch image
        if 'pitch_image' not in request.files:
            return jsonify({"success": False, "error": "No pitch image provided"})
            
        pitch_file = request.files['pitch_image']
        pitch_path = os.path.join(UPLOAD_FOLDER, f"{video_id}_pitch.jpg")
        pitch_file.save(pitch_path)
        
        # Get the calibration data
        pitch_length = float(request.form.get('pitch_length', 22.0))  # Default cricket pitch length in meters
        pitch_width = float(request.form.get('pitch_width', 3.05))    # Default cricket pitch width in meters
        
        # Load the pitch image
        pitch_img = cv2.imread(pitch_path)
        if pitch_img is None:
            return jsonify({"success": False, "error": "Failed to load pitch image"})
            
        # Get image dimensions
        img_height, img_width = pitch_img.shape[:2]
        
        # Get the video dimensions
        video_width = tracking_data["video_info"]["width"]
        video_height = tracking_data["video_info"]["height"]
        
        # Scale factor for mapping video coordinates to pitch image
        scale_x = img_width / video_width
        scale_y = img_height / video_height
        
        # Create visualization for each ball trajectory
        paths = []
        for ball_idx, ball_path in enumerate(tracking_data["balls"]):
            # Skip if too few points
            if len(ball_path) < 2:
                continue
                
            # Create a copy of the pitch image for this path
            path_img = pitch_img.copy()
            
            # Draw the path on the image
            points = []
            for i in range(len(ball_path)):
                # Map video coordinates to pitch image coordinates
                x = int(ball_path[i][0] * scale_x)
                y = int(ball_path[i][1] * scale_y )
                points.append((x, y))
                
                # Draw point
                cv2.circle(path_img, (x, y), 5, (0, 0, 255), -1)
                
                # Connect points with a line
                if i > 0:
                    cv2.line(path_img, points[i-1], points[i], (255, 0, 0), 2)
            
            # Calculate the actual distance traveled
            total_pixels = 0
            for i in range(1, len(points)):
                dx = points[i][0] - points[i-1][0]
                dy = points[i][1] - points[i-1][1]
                total_pixels += np.sqrt(dx*dx + dy*dy)
            
            # Calculate the pixels per meter based on the pitch dimensions
            # Assuming pitch length is along the y-axis and width along the x-axis
            pixels_per_meter_x = img_width / pitch_width
            pixels_per_meter_y = img_height / pitch_length
            
            # Use the average pixels per meter for distance calculation
            pixels_per_meter = (pixels_per_meter_x + pixels_per_meter_y) / 2
            
            # Calculate distance in meters
            distance_meters = total_pixels / pixels_per_meter
            
            # Add the distance information to the image
            text = f"Distance: {distance_meters:.2f} meters"
            cv2.putText(path_img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            # Save the visualization
            output_path = os.path.join(RESULTS_FOLDER, f"{video_id}_path_{ball_idx}.jpg")
            cv2.imwrite(output_path, path_img)
            
            # Add path info to the result
            paths.append({
                "path_id": ball_idx,
                "image": f"{video_id}_path_{ball_idx}.jpg",
                "distance": distance_meters,
                "points": points
            })
            
        # Return the path information
        return jsonify({
            "success": True, 
            "paths": paths
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# Add a route to retrieve the path visualizations
@app.route('/path_image/<path_id>', methods=['GET'])
def get_path_image(path_id):
    return send_from_directory(RESULTS_FOLDER, path_id)

# Add calibration functionality
@app.route('/calibrate', methods=['POST'])
def calibrate_video():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No video file provided"})
            
        video_file = request.files['file']
        
        # Save the calibration video temporarily
        calib_id = str(uuid.uuid4())
        _, ext = os.path.splitext(video_file.filename)
        calib_path = os.path.join(UPLOAD_FOLDER, f"{calib_id}{ext}")
        video_file.save(calib_path)
        
        # Get calibration settings
        real_length = float(request.form.get('real_length', 22.0))  # Real-world length in meters
        
        # Open the video
        cap = cv2.VideoCapture(calib_path)
        if not cap.isOpened():
            return jsonify({"success": False, "error": "Failed to open calibration video"})
        
        # Get the first frame for calibration
        ret, frame = cap.read()
        if not ret:
            return jsonify({"success": False, "error": "Failed to read frame from video"})
            
        # Save the frame for calibration
        calib_frame_path = os.path.join(RESULTS_FOLDER, f"{calib_id}_frame.jpg")
        cv2.imwrite(calib_frame_path, frame)
        
        # Close the video
        cap.release()
        
        # Return the calibration frame path for the user to mark points
        return jsonify({
            "success": True,
            "calibration_id": calib_id,
            "frame_image": f"{calib_id}_frame.jpg",
            "message": "Please mark two points on the image to set the reference length"
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# Process the calibration points
@app.route('/process_calibration', methods=['POST'])
def process_calibration():
    try:
        data = request.json
        calib_id = data.get('calibration_id')
        point1 = data.get('point1')  # [x, y]
        point2 = data.get('point2')  # [x, y]
        real_length = float(data.get('real_length', 22.0))  # Real-world distance in meters
        
        # Calculate pixel distance
        pixel_distance = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
        
        # Calculate pixels per meter
        pixels_per_meter = pixel_distance / real_length
        
        # Save the calibration information
        calib_info = {
            "pixels_per_meter": pixels_per_meter,
            "real_length": real_length,
            "point1": point1,
            "point2": point2
        }
        
        calib_file = os.path.join(RESULTS_FOLDER, f"{calib_id}_calib.json")
        with open(calib_file, 'w') as f:
            json.dump(calib_info, f)
        
        return jsonify({
            "success": True,
            "calibration_id": calib_id,
            "pixels_per_meter": pixels_per_meter
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)