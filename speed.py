from collections import deque
from ultralytics import YOLO
import math
import time
import cv2
import os
import json
import numpy as np

def angle_between_lines(m1, m2=1):
    if m1 != -1/m2:
        angle = math.degrees(math.atan(abs((m2 - m1) / (1 + m1 * m2))))
        return angle
    else:
        return 90.0

class FixedSizeQueue:
    def __init__(self, max_size):
        self.queue = deque(maxlen=max_size)
    
    def add(self, item):
        self.queue.append(item)
    
    def pop(self):
        self.queue.popleft()
        
    def clear(self):
        self.queue.clear()

    def get_queue(self):
        return self.queue
    
    def __len__(self):
        return len(self.queue)

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

def process_video(video_path, output_json_path, model_path):
    """Process video once and save all tracking data to JSON file"""
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Dictionary to store all tracking data
    tracking_data = {
        "video_info": {
            "fps": fps,
            "frame_count": frame_count,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        },
        "frames": {}
    }
    
    frame_idx = 0
    centroid_history = FixedSizeQueue(10)
    prev_centroid = None
    
    # For speed calibration - you'll need to adjust this based on your video
    pixels_per_meter = 100  # This value needs to be calibrated!
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame with YOLO
        results = model.track(frame, persist=True, conf=0.35, verbose=False)
        boxes = results[0].boxes
        box = boxes.xyxy
        
        # Data for this frame
        frame_data = {
            "centroids": [],
            "bounding_boxes": [],
            "future_positions": [],
            "angle": 0,
            "speed": 0
        }
        
        current_centroid = None
        
        if len(box) != 0:
            rows, cols = box.shape
            for i in range(rows):
                x1, y1, x2, y2 = box[i]
                x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
                
                centroid_x = int((x1+x2)/2)
                centroid_y = int((y1+y2)/2)
                current_centroid = (centroid_x, centroid_y)
                
                centroid_history.add(current_centroid)
                frame_data["centroids"].append([centroid_x, centroid_y])
                frame_data["bounding_boxes"].append([int(x1), int(y1), int(x2), int(y2)])
        
        # Calculate speed
        if prev_centroid is not None and current_centroid is not None:
            speed = calculate_speed(prev_centroid, current_centroid, fps, pixels_per_meter)
            frame_data["speed"] = speed
        
        if current_centroid is not None:
            prev_centroid = current_centroid
        
        # Calculate angles and future positions
        angle = 0
        if len(centroid_history) > 1:
            centroid_list = list(centroid_history.get_queue())
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
        
        # Update for next iteration
        frame_idx += 1
        
        # Show progress
        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx}/{frame_count} frames ({(frame_idx/frame_count*100):.1f}%)")
    
    # Save all tracking data to file
    with open(output_json_path, 'w') as f:
        json.dump(tracking_data, f)
    
    cap.release()
    print(f"Processing complete. Data saved to {output_json_path}")

def replay_with_overlay(video_path, tracking_data_path):
    """Replay the video with overlaid tracking information"""
    # Load tracking data
    with open(tracking_data_path, 'r') as f:
        tracking_data = json.load(f)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    video_fps = tracking_data["video_info"]["fps"]
    frame_delay = int(1000 / video_fps)  # milliseconds between frames for natural speed
    
    frame_idx = 0
    paused = False
    
    while True:
        # Capture current time to maintain consistent framerate
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get tracking data for this frame
        frame_data = tracking_data["frames"].get(str(frame_idx), {})
        
        # Draw centroids and bounding boxes
        for centroid in frame_data.get("centroids", []):
            cv2.circle(frame, (centroid[0], centroid[1]), radius=3, color=(0, 0, 255), thickness=-1)
        
        for box in frame_data.get("bounding_boxes", []):
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        
        # Display speed
        speed = frame_data.get("speed", 0)
        print(speed,"spped")
        if speed > 0:
            speed_text = "Speed: {:.1f} km/h".format(speed)
            cv2.putText(frame, speed_text, (20, 80), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        
        # Show frame
        cv2.imshow('frame', frame)
        
        # Calculate how much time we need to wait to maintain proper framerate
        processing_time = (time.time() - start_time) * 1000  # convert to ms
        wait_time = max(1, int(frame_delay - processing_time))  # ensure at least 1ms
        
        # Handle keyboard input with proper timing
        key = cv2.waitKey(wait_time) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Space to pause/resume
            paused = not paused
            while paused:
                key = cv2.waitKey(30) & 0xFF
                if key == ord(' '):
                    paused = not paused
                elif key == ord('q'):
                    break
        
        if not paused:
            frame_idx += 1
    
    cap.release()
    cv2.destroyAllWindows()
def calibrate_pixels_per_meter(video_path):
    """
    Interactive calibration function to determine pixels per meter
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        cap.release()
        return None
    
    # Resize for display
    frame_resized = cv2.resize(frame, (1000, 600))
    
    # Instructions
    print("CALIBRATION: Click two points with a known distance between them")
    print("For cricket, you might use the distance between stumps (9 inches) or pitch markings")
    
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert coordinates back to original frame size
            h_ratio = frame.shape[0] / frame_resized.shape[0]
            w_ratio = frame.shape[1] / frame_resized.shape[1]
            orig_x = int(x * w_ratio)
            orig_y = int(y * h_ratio)
            
            points.append((orig_x, orig_y))
            cv2.circle(frame_resized, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Calibration', frame_resized)
            
            if len(points) == 2:
                pixel_distance = math.sqrt((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2)
                print(f"Pixel distance between points: {pixel_distance}")
                cv2.line(frame_resized, 
                         (int(points[0][0]/w_ratio), int(points[0][1]/h_ratio)), 
                         (int(points[1][0]/w_ratio), int(points[1][1]/h_ratio)), 
                         (0, 0, 255), 2)
                cv2.imshow('Calibration', frame_resized)
                
    cv2.namedWindow('Calibration')
    cv2.setMouseCallback('Calibration', mouse_callback)
    cv2.imshow('Calibration', frame_resized)
    
    while len(points) < 2:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Calculate pixels per meter based on user input
    if len(points) == 2:
        pixel_distance = math.sqrt((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2)
        
        # Ask user for the real-world distance
        real_distance = input("Enter the real-world distance between these points in meters: ")
        try:
            real_distance = float(real_distance)
            pixels_per_meter = pixel_distance / real_distance
            print(f"Calibration complete: {pixels_per_meter} pixels = 1 meter")
        except ValueError:
            print("Invalid input. Using default value of 100 pixels per meter.")
            pixels_per_meter = 100
    else:
        pixels_per_meter = None
    
    cap.release()
    cv2.destroyAllWindows()
    return pixels_per_meter
if __name__ == "__main__":
    video_path = os.path.join('videos', 'test1.mp4')
    model_path = os.path.join('runs', 'detect', 'train5', 'weights', 'best.pt')
    tracking_data_path = 'ball_tracking_data.json'
    
    # First, calibrate the video
    pixels_per_meter = calibrate_pixels_per_meter(video_path)
    if pixels_per_meter is None:
        print("Using default calibration of 100 pixels per meter")
        pixels_per_meter = 100
    
    # # Then process the video with the calibrated value
    # # Uncomment the one you want to run:
    process_video(video_path, tracking_data_path, model_path)  # First process and save data
    replay_with_overlay(video_path, tracking_data_path)  # Then replay with overlay