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

def process_video(video_path, output_json_path, model_path, pixels_per_meter=100):
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
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "pixels_per_meter": pixels_per_meter
        },
        "frames": {},
        "balls": []  # List to store separate ball trajectories
    }
    
    frame_idx = 0
    centroid_history = FixedSizeQueue(10)
    no_detection_count = 0
    current_ball_centroids = []
    
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
            "angle": 0
        }
        
        # Check if we detected a ball in this frame
        if len(box) != 0:
            rows, cols = box.shape
            for i in range(rows):
                x1, y1, x2, y2 = box[i]
                x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
                
                centroid_x = int((x1+x2)/2)
                centroid_y = int((y1+y2)/2)
                
                centroid_history.add((centroid_x, centroid_y))
                frame_data["centroids"].append([centroid_x, centroid_y])
                frame_data["bounding_boxes"].append([int(x1), int(y1), int(x2), int(y2)])
                
                # Add to current ball centroids - now include frame index for timing
                current_ball_centroids.append([centroid_x, centroid_y, frame_idx])
            
            # Reset no detection counter since we found a ball
            no_detection_count = 0
        else:
            # Increment no detection counter
            no_detection_count += 1
            
            # If we've had multiple frames with no detection and we have data for the current ball,
            # save this ball's trajectory and prepare for the next ball
            if no_detection_count > 15 and len(current_ball_centroids) > 0:
                tracking_data["balls"].append(current_ball_centroids)
                current_ball_centroids = []
                centroid_history.clear()
        
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
    
    # Add the last ball trajectory if it exists
    if len(current_ball_centroids) > 0:
        tracking_data["balls"].append(current_ball_centroids)
    
    # Calculate speeds for all balls
    for ball_idx, ball in enumerate(tracking_data["balls"]):
        # Add speed information to each ball
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
            
        # Update the ball data with speed information
        tracking_data["balls"][ball_idx] = ball_with_speed
    
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
    pixels_per_meter = tracking_data["video_info"].get("pixels_per_meter", 100)
    
    frame_idx = 0
    paused = False
    
    # Extract ball trajectories
    all_balls = tracking_data["balls"]
    
    # List to store active balls (balls currently visible in the frame)
    active_balls = []
    ball_speeds = {}  # Dictionary to store current speed of each ball
    
    # Color map for different balls
    ball_colors = [
        (0, 0, 255),     # Red
        (0, 255, 0),     # Green
        (255, 0, 0),     # Blue
        (0, 255, 255),   # Yellow
        (255, 0, 255),   # Magenta
        (255, 255, 0),   # Cyan
        (128, 0, 0),     # Dark Blue
        (0, 128, 0),     # Dark Green
        (0, 0, 128),     # Dark Red
        (128, 128, 0)    # Dark Cyan
    ]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get tracking data for this frame
        frame_data = tracking_data["frames"].get(str(frame_idx), {})
        
        # Clear active balls
        active_balls = []
        ball_speeds = {}
        
        # Check for balls active in this frame
        for ball_idx, ball in enumerate(all_balls):
            # Get frame indices for this ball
            frame_indices = [c[2] for c in ball]
            
            # If this ball appears in the current frame
            if frame_idx in frame_indices:
                active_balls.append((ball_idx, ball))
                
                # Find the position in the ball data for the current frame
                pos_idx = frame_indices.index(frame_idx)
                
                # Get the speed for this frame if available (at index 3)
                if len(ball[pos_idx]) > 3:
                    ball_speeds[ball_idx] = ball[pos_idx][3]
                else:
                    ball_speeds[ball_idx] = 0.0
            
            # If this ball appears within 5 frames of the current frame, show it as active too
            # This helps with visualization when there are temporary detection failures
            elif any(abs(fi - frame_idx) <= 5 for fi in frame_indices):
                active_balls.append((ball_idx, ball))
                # Find the nearest frame
                nearest_frame_idx = min(frame_indices, key=lambda x: abs(x - frame_idx))
                pos_idx = frame_indices.index(nearest_frame_idx)
                if len(ball[pos_idx]) > 3:
                    ball_speeds[ball_idx] = ball[pos_idx][3]
                else:
                    ball_speeds[ball_idx] = 0.0
        
        # Draw each active ball's path
        for ball_idx, ball in active_balls:
            color = ball_colors[ball_idx % len(ball_colors)]
            
            # Draw the ball's path up to this frame
            centroids_up_to_frame = [c for c in ball if c[2] <= frame_idx]
            if len(centroids_up_to_frame) > 1:
                # Create a polyline of all previous positions
                path_points = [(c[0], c[1]) for c in centroids_up_to_frame]
                path_array = np.array(path_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [path_array], False, color, 2)
                
                # Draw the current position
                cv2.circle(frame, (path_points[-1][0], path_points[-1][1]), 
                         radius=5, color=color, thickness=-1)
                
                # Display the ball's speed
                speed = ball_speeds.get(ball_idx, 0.0)
                if speed > 0:
                    speed_text = f"Ball {ball_idx+1}: {speed:.1f} km/h"
                    text_x = path_points[-1][0] + 10
                    text_y = path_points[-1][1] + 10
                    cv2.putText(frame, speed_text, (text_x, text_y), 
                              cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
        
        # Draw centroids and bounding boxes for current frame
        for centroid in frame_data.get("centroids", []):
            cv2.circle(frame, (centroid[0], centroid[1]), radius=5, color=(0, 0, 255), thickness=-1)
        
        for box in frame_data.get("bounding_boxes", []):
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        
        # Draw future position lines
        future_positions = frame_data.get("future_positions", [])
        if len(future_positions) > 0:
            for i in range(1, len(future_positions)):
                cv2.line(frame, 
                       (int(future_positions[i-1][0]), int(future_positions[i-1][1])), 
                       (int(future_positions[i][0]), int(future_positions[i][1])), 
                       (0, 255, 0), 2)
                cv2.circle(frame, 
                         (int(future_positions[i][0]), int(future_positions[i][1])), 
                         radius=3, color=(0, 255, 0), thickness=-1)
        
        # Display angle
        angle = frame_data.get("angle", 0)
        text = f"Angle: {angle:.2f} degrees"
        cv2.putText(frame, text, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        
        # Display FPS and frame number
        cv2.putText(frame, f'FPS: {video_fps} | Frame: {frame_idx}', (20, 50), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Display number of active balls
        cv2.putText(frame, f'Active balls: {len(active_balls)}', (20, 80), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Show frame
        cv2.imshow('Ball Tracking with Speed', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(frame_delay) & 0xFF
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

def analyze_ball_speeds(tracking_data_path):
    """Analyze and display statistics about ball speeds"""
    with open(tracking_data_path, 'r') as f:
        tracking_data = json.load(f)
    
    all_balls = tracking_data["balls"]
    
    print("\n=== Ball Speed Analysis ===")
    print(f"Number of detected balls: {len(all_balls)}")
    
    for ball_idx, ball in enumerate(all_balls):
        speeds = [point[3] for point in ball if len(point) > 3]
        if speeds:
            max_speed = max(speeds)
            avg_speed = sum(speeds) / len(speeds)
            print(f"\nBall #{ball_idx+1}:")
            print(f"  Number of tracked positions: {len(ball)}")
            print(f"  Maximum speed: {max_speed:.2f} km/h")
            print(f"  Average speed: {avg_speed:.2f} km/h")
            
            # Calculate frame range
            frame_start = ball[0][2]
            frame_end = ball[-1][2]
            print(f"  Frame range: {frame_start} - {frame_end} ({frame_end - frame_start + 1} frames)")

if __name__ == "__main__":
    video_path = os.path.join('videos', '7.mp4')
    model_path = os.path.join('runs', 'detect', 'train9', 'weights', 'best.pt')
    tracking_data_path = 'ball_tracking_data.json'
    
    # First, calibrate the video
    pixels_per_meter=None
    pixels_per_meter = calibrate_pixels_per_meter(video_path)
    if pixels_per_meter is None:
        print("Using default calibration of 100 pixels per meter")
        pixels_per_meter = 100
    
    # # Process the video with the calibrated value
    process_video(video_path, tracking_data_path, model_path, pixels_per_meter)
    
    # Analyze ball speeds
    analyze_ball_speeds(tracking_data_path)
    
    # Replay with overlay
    replay_with_overlay(video_path, tracking_data_path)