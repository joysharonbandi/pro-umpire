import cv2
import numpy as np
import math
import json
import time
from ultralytics import YOLO  # Make sure to install with: pip install ultralytics

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def calculate_angle(point1, point2):
    """Calculate angle between two points (in degrees)"""
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    return math.degrees(math.atan2(dy, dx))

def predict_points(last_point, angle, distance=50, points=10):
    """Predict future points based on angle and distance"""
    predicted_points = []
    
    # Convert angle to radians
    angle_rad = math.radians(angle)
    
    # Calculate step size for smooth prediction
    step = distance / points
    
    for i in range(1, points + 1):
        # Calculate offset distance
        offset_distance = step * i
        
        # Calculate new x and y coordinates
        x = last_point[0] + offset_distance * math.cos(angle_rad)
        y = last_point[1] + offset_distance * math.sin(angle_rad)
        
        predicted_points.append((int(x), int(y)))
    
    return predicted_points

def get_rectangle_corners_from_flat(bbox):
    """Convert flat bbox [x1, y1, x2, y2] to corner points [(x1,y1), (x2,y1), (x2,y2), (x1,y2)]"""
    x1, y1, x2, y2 = bbox
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

def is_point_in_polygon(vertices, point):
    """Check if a point is inside a polygon using ray casting algorithm"""
    x, y = point
    n = len(vertices)
    inside = False
    
    p1x, p1y = vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = vertices[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def draw_ball_trajectory(tracking_data, display_frame, current_frame_idx):
    """Draw the trajectory of the ball based on previous frames"""
    # Get past frames to determine trajectory
    max_trajectory_length = 20  # Increased for more visible trajectory
    start_frame = max(0, current_frame_idx - max_trajectory_length)
    
    ball_positions = []
    for i in range(start_frame, current_frame_idx):
        if i < len(tracking_data["frames"]):
            frame_data = tracking_data["frames"][i]
            if "ball" in frame_data["objects"] and frame_data["objects"]["ball"]:
                # Use the first detected ball
                ball_pos = tuple(map(int, frame_data["objects"]["ball"][0]["center"]))
                ball_positions.append(ball_pos)
    
    # Check if we have enough ball positions for trajectory
    if len(ball_positions) <= 1:
        return False, None, None
    
    # Current frame data
    current_frame_data = tracking_data["frames"][current_frame_idx-1]
    
    # Check if ball is near batsman or bat (impact detection)
    impact_detected = False
    impact_point = None
    
    if "distances" in current_frame_data:
        if "ball_to_batsman" in current_frame_data["distances"] and current_frame_data["distances"]["ball_to_batsman"] < 100:
            impact_detected = True
            if "ball" in current_frame_data["objects"] and current_frame_data["objects"]["ball"]:
                impact_point = tuple(map(int, current_frame_data["objects"]["ball"][0]["center"]))
        elif "ball_to_bat" in current_frame_data["distances"] and current_frame_data["distances"]["ball_to_bat"] < 50:
            impact_detected = True
            if "ball" in current_frame_data["objects"] and current_frame_data["objects"]["ball"]:
                impact_point = tuple(map(int, current_frame_data["objects"]["ball"][0]["center"]))
    
    # Draw trajectory lines
    if len(ball_positions) > 1:
        # Calculate trajectory angle for the last few positions
        if len(ball_positions) >= 3:
            last_three_positions = ball_positions[-3:]
            pre_impact_angle = calculate_angle(last_three_positions[0], last_three_positions[-1])
        else:
            pre_impact_angle = calculate_angle(ball_positions[0], ball_positions[-1])
        
        # Draw actual trajectory
        for i in range(1, len(ball_positions)):
            cv2.line(display_frame, ball_positions[i-1], ball_positions[i], (0, 255, 255), 3)
        
        # Mark the current ball position
        if ball_positions:
            cv2.circle(display_frame, ball_positions[-1], 7, (0, 0, 255), -1)
    
    return impact_detected, impact_point, pre_impact_angle

def predict_future_trajectory(impact_point, pre_impact_angle, display_frame):
    """Predict and draw the future trajectory of the ball after impact"""
    # Reflect the angle based on impact type (simplified physics)
    # In a real system, we'd use more complex physics based on impact type
    post_impact_angle = -pre_impact_angle + 180  # Simple reflection
    
    # Predict future points
    future_points = predict_points(impact_point, post_impact_angle, distance=300, points=15)
    
    # Draw predicted trajectory
    for i in range(1, len(future_points)):
        cv2.line(display_frame, future_points[i-1], future_points[i], (255, 0, 255), 3)  # Magenta for predicted path
    
    # Add a different colored circle at the end of prediction
    if future_points:
        cv2.circle(display_frame, future_points[-1], 7, (255, 0, 255), -1)
    
    return future_points

def display_phase_info(display_frame, current_frame_number, has_impact=False):
    """Display the current phase of the ball trajectory"""
    if has_impact:
        phase_text = "Post-Impact Prediction"
    elif 90 <= current_frame_number < 98:
        phase_text = "Pre-Delivery"
    elif 98 <= current_frame_number < 104:
        phase_text = "Ball Delivery"
    elif 104 <= current_frame_number < 110:
        phase_text = "Ball Approaching Batsman"
    elif 110 <= current_frame_number < 117:
        phase_text = "Ball Settling"
    else:
        phase_text = "Ball Tracking"
    
    # Add phase information at the bottom of the frame
    h, w = display_frame.shape[:2]
    cv2.rectangle(display_frame, (10, h-40), (350, h-10), (0, 0, 0), -1)
    cv2.putText(display_frame, f"Phase: {phase_text}", (20, h-20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def display_distance_measurements(display_frame, current_frame_data):
    """Display the distance measurements on the frame"""
    # Create a panel for distance information
    cv2.rectangle(display_frame, (10, 10), (350, 130), (0, 0, 0), -1)
    cv2.rectangle(display_frame, (10, 10), (350, 130), (255, 255, 255), 2)
    
    # Add title
    cv2.putText(display_frame, "Distance Measurements (pixels)", (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display Ball to Bowler distance
    if "distances" in current_frame_data and "ball_to_bowler" in current_frame_data["distances"]:
        dist = current_frame_data["distances"]["ball_to_bowler"]
        cv2.putText(display_frame, f"Ball to Bowler: {dist:.1f}", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Color-coded indicator
        if dist < 500:
            cv2.circle(display_frame, (330, 65), 8, (0, 255, 0), -1)  # Green when close
        else:
            cv2.circle(display_frame, (330, 65), 8, (0, 0, 255), -1)  # Red when far
    
    # Display Ball to Batsman distance
    if "distances" in current_frame_data and "ball_to_batsman" in current_frame_data["distances"]:
        dist = current_frame_data["distances"]["ball_to_batsman"]
        cv2.putText(display_frame, f"Ball to Batsman: {dist:.1f}", (20, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Color-coded indicator
        if dist < 100:
            cv2.circle(display_frame, (330, 95), 8, (0, 255, 0), -1)  # Green when close
        else:
            cv2.circle(display_frame, (330, 95), 8, (0, 0, 255), -1)  # Red when far
    
    # Display Ball to Bat distance
    if "distances" in current_frame_data and "ball_to_bat" in current_frame_data["distances"]:
        dist = current_frame_data["distances"]["ball_to_bat"]
        cv2.putText(display_frame, f"Ball to Bat: {dist:.1f}", (20, 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Color-coded indicator
        if dist < 50:
            cv2.circle(display_frame, (330, 125), 8, (0, 255, 0), -1)  # Green when close
        else:
            cv2.circle(display_frame, (330, 125), 8, (0, 0, 255), -1)  # Red when far

def calculate_distances(frame_data):
    """Calculate distances between different objects in the frame"""
    # Initialize distances dictionary if not present
    if "distances" not in frame_data:
        frame_data["distances"] = {}
    
    objects = frame_data["objects"]
    
    # Ball to Bowler Hand distance
    if "ball" in objects and "bowler_hand" in objects:
        for ball in objects["ball"]:
            for bowler_hand in objects["bowler_hand"]:
                distance = calculate_distance(ball["center"], bowler_hand["center"])
                frame_data["distances"]["ball_to_bowler_hand"] = distance
    
    # Ball to Bowler distance
    if "ball" in objects and "bowler" in objects:
        for ball in objects["ball"]:
            for bowler in objects["bowler"]:
                distance = calculate_distance(ball["center"], bowler["center"])
                frame_data["distances"]["ball_to_bowler"] = distance
    
    # Ball to Batsman distance
    if "ball" in objects and "batsman" in objects:
        for ball in objects["ball"]:
            for batsman in objects["batsman"]:
                distance = calculate_distance(ball["center"], batsman["center"])
                frame_data["distances"]["ball_to_batsman"] = distance
    
    # Ball to Bat distance
    if "ball" in objects and "bat" in objects:
        for ball in objects["ball"]:
            for bat in objects["bat"]:
                distance = calculate_distance(ball["center"], bat["center"])
                frame_data["distances"]["ball_to_bat"] = distance

def predict_cricket_video(video_path, model_path):
    """Main function to predict ball trajectory in cricket video"""
    # Initialize variables
    paused = False
    impact_detected = False
    impact_handled = False
    future_trajectory = None
    
    # Load YOLO model
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create writer for output video
    output_path = 'output_prediction.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize tracking data structure
    tracking_data = {
        "video_info": {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height
        },
        "frames": []
    }
    
    # Calculate frame delay for display
    frame_delay = int(1000 / fps) if fps > 0 else 30
    
    # Process video frames
    current_frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create a copy for display
        display_frame = frame.copy()
        
        # Run YOLOv8 detection
        results = model(frame)
        
        # Initialize frame data
        frame_data = {
            "frame_number": current_frame_idx,
            "objects": {},
            "distances": {}
        }
        
        # Process detection results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get class information
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                
                # Calculate center point
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Store object information
                if class_name not in frame_data["objects"]:
                    frame_data["objects"][class_name] = []
                
                frame_data["objects"][class_name].append({
                    "bbox": [x1, y1, x2, y2],
                    "center": [center_x, center_y],
                    "confidence": confidence
                })
                
                # Draw bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"{class_name} {confidence:.2f}", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Calculate distances between objects
        calculate_distances(frame_data)
        
        # Store frame data
        tracking_data["frames"].append(frame_data)
        
        # Draw trajectory if we have enough frames
        if current_frame_idx > 0:
            # Only process impact once
            if not impact_handled:
                impact_detected, impact_point, pre_impact_angle = draw_ball_trajectory(
                    tracking_data, display_frame, current_frame_idx)
                
                # If impact is detected, predict future trajectory
                if impact_detected and impact_point and pre_impact_angle:
                    future_trajectory = predict_future_trajectory(
                        impact_point, pre_impact_angle, display_frame)
                    impact_handled = True
            else:
                # Continue drawing past trajectory
                draw_ball_trajectory(tracking_data, display_frame, current_frame_idx)
                
                # Draw the saved future trajectory
                if future_trajectory and len(future_trajectory) > 1:
                    for i in range(1, len(future_trajectory)):
                        cv2.line(display_frame, future_trajectory[i-1], future_trajectory[i], 
                                (255, 0, 255), 3)
                    cv2.circle(display_frame, future_trajectory[-1], 7, (255, 0, 255), -1)
        
        # Display phase information
        display_phase_info(display_frame, current_frame_idx, impact_handled)
        
        # Display distance measurements
        if current_frame_idx > 0:
            display_distance_measurements(display_frame, frame_data)
        
        # Add frame number
        cv2.putText(display_frame, f"Frame: {current_frame_idx}", 
                    (width - 150, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow("Cricket Ball Trajectory Analysis", display_frame)
        
        # Write frame to output video
        out.write(display_frame)
        
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
        
        current_frame_idx += 1
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Save tracking data to file
    with open('ball_tracking_data.json', 'w') as f:
        json.dump(tracking_data, f, indent=2)
    
    print(f"Analysis complete. Output video saved to {output_path}")
    print(f"Tracking data saved to ball_tracking_data.json")
    
    return tracking_data

if __name__ == "__main__":
    import os
    
    # Define paths for video and model
    video_path = os.path.join('videos', 'ball.mov')
    model_path = os.path.join('runs', 'detect', 'train11', 'best.pt')
    
    # Execute the main function
    predict_cricket_video(video_path, model_path)