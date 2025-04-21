# from collections import deque
# from ultralytics import YOLO
# import math
# import time
# import cv2
# import os
# import json
# import numpy as np

# def angle_between_lines(m1, m2=1):
#     if m1 != -1/m2:
#         angle = math.degrees(math.atan(abs((m2 - m1) / (1 + m1 * m2))))
#         return angle
#     else:
#         return 90.0

# class FixedSizeQueue:
#     def __init__(self, max_size):
#         self.queue = deque(maxlen=max_size)
    
#     def add(self, item):
#         self.queue.append(item)
    
#     def pop(self):
#         self.queue.popleft()
        
#     def clear(self):
#         self.queue.clear()

#     def get_queue(self):
#         return self.queue
    
#     def __len__(self):
#         return len(self.queue)

# def process_video(video_path, output_json_path, model_path):
#     """Process video once and save all tracking data to JSON file"""
#     model = YOLO(model_path)
#     cap = cv2.VideoCapture(video_path)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
    
#     # Dictionary to store all tracking data
#     tracking_data = {
#         "video_info": {
#             "fps": fps,
#             "frame_count": frame_count,
#             "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
#             "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         },
#         "frames": {},
#         "all_centroids": []  # Store all centroids across all frames
#     }
    
#     frame_idx = 0
#     centroid_history = FixedSizeQueue(10)
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         # Process frame with YOLO
#         results = model.track(frame, persist=True, conf=0.35, verbose=False)
#         boxes = results[0].boxes
#         box = boxes.xyxy
        
#         # Data for this frame
#         frame_data = {
#             "centroids": [],
#             "bounding_boxes": [],
#             "future_positions": [],
#             "angle": 0
#         }
        
#         if len(box) != 0:
#             rows, cols = box.shape
#             for i in range(rows):
#                 x1, y1, x2, y2 = box[i]
#                 x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
                
#                 centroid_x = int((x1+x2)/2)
#                 centroid_y = int((y1+y2)/2)
                
#                 centroid_history.add((centroid_x, centroid_y))
#                 tracking_data["all_centroids"].append([frame_idx, centroid_x, centroid_y])  # Store frame index with centroid
#                 frame_data["centroids"].append([centroid_x, centroid_y])
#                 frame_data["bounding_boxes"].append([int(x1), int(y1), int(x2), int(y2)])
        
#         # Calculate angles and future positions
#         angle = 0
#         if len(centroid_history) > 1:
#             centroid_list = list(centroid_history.get_queue())
#             x_diff = centroid_list[-1][0] - centroid_list[-2][0]
#             y_diff = centroid_list[-1][1] - centroid_list[-2][1]
            
#             if x_diff != 0:
#                 m1 = y_diff/x_diff
#                 if m1 == 1:
#                     angle = 90
#                 elif m1 != 0:
#                     angle = 90-angle_between_lines(m1)
            
#             future_positions = [centroid_list[-1]]
#             for i in range(1, 5):
#                 future_pos = (
#                     centroid_list[-1][0] + x_diff * i,
#                     centroid_list[-1][1] + y_diff * i
#                 )
#                 future_positions.append(future_pos)
#                 frame_data["future_positions"].append([int(future_pos[0]), int(future_pos[1])])
            
#             frame_data["angle"] = angle
        
#         # Store frame data
#         tracking_data["frames"][frame_idx] = frame_data
        
#         # Update for next iteration
#         frame_idx += 1
        
#         # Show progress
#         if frame_idx % 10 == 0:
#             print(f"Processed {frame_idx}/{frame_count} frames ({(frame_idx/frame_count*100):.1f}%)")
    
#     # Save all tracking data to file
#     with open(output_json_path, 'w') as f:
#         json.dump(tracking_data, f)
    
#     cap.release()
#     print(f"Processing complete. Data saved to {output_json_path}")

# def replay_with_overlay(video_path, tracking_data_path):
#     """Replay the video with overlaid tracking information"""
#     # Load tracking data
#     with open(tracking_data_path, 'r') as f:
#         tracking_data = json.load(f)
    
#     # Open video
#     cap = cv2.VideoCapture(video_path)
#     video_fps = tracking_data["video_info"]["fps"]
#     frame_delay = int(1000 / video_fps)  # milliseconds between frames for natural speed
    
#     frame_idx = 0
#     paused = False
    
#     # Extract all centroids for full path display
#     all_centroids = tracking_data.get("all_centroids", [])
    
#     # Color map for trajectory visualization - changes color gradually based on frame
#     if all_centroids:
#         max_frame = max([c[0] for c in all_centroids])
        
#     # Track visible path length
#     path_length = 0  # 0 means show all, positive number limits path length
#     show_full_path = True
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Get tracking data for this frame
#         frame_data = tracking_data["frames"].get(str(frame_idx), {})
        
#         # Draw full path or partial path based on settings
#         if show_full_path:
#             # Draw all points up to current frame
#             visible_centroids = [c for c in all_centroids if c[0] <= frame_idx]
#         elif path_length > 0:
#             # Draw only the recent points within the path_length
#             visible_centroids = [c for c in all_centroids if c[0] <= frame_idx and c[0] > frame_idx - path_length]
#         else:
#             visible_centroids = []
            
#         # Draw the path with color gradient
#         if len(visible_centroids) > 1:
#             for i in range(1, len(visible_centroids)):
#                 prev_frame, prev_x, prev_y = visible_centroids[i-1]
#                 curr_frame, curr_x, curr_y = visible_centroids[i]
                
#                 # Skip if frames are not consecutive (for better visualization)
#                 if curr_frame - prev_frame > 5:
#                     continue
                
#                 # Create color gradient based on frame position
#                 if max_frame > 0:
#                     # Gradient from blue (old) to red (recent)
#                     ratio = curr_frame / max_frame
#                     color = (
#                         int(255 * ratio),        # B component
#                         0,                        # G component
#                         int(255 * (1 - ratio))   # R component
#                     )
#                 else:
#                     color = (0, 255, 0)  # Default green if no frames
                
#                 cv2.line(frame, (prev_x, prev_y), (curr_x, curr_y), color, 2)
        
#         # Draw centroids and bounding boxes for current frame
#         for centroid in frame_data.get("centroids", []):
#             cv2.circle(frame, (centroid[0], centroid[1]), radius=5, color=(0, 0, 255), thickness=-1)
        
#         for box in frame_data.get("bounding_boxes", []):
#             cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        
#         # Draw future position lines
#         future_positions = frame_data.get("future_positions", [])
#         if len(future_positions) > 0:
#             for i in range(1, len(future_positions)):
#                 cv2.line(frame, 
#                        (int(future_positions[i-1][0]), int(future_positions[i-1][1])), 
#                        (int(future_positions[i][0]), int(future_positions[i][1])), 
#                        (0, 255, 0), 4)
#                 cv2.circle(frame, 
#                          (int(future_positions[i][0]), int(future_positions[i][1])), 
#                          radius=3, color=(0, 0, 255), thickness=-1)
        
#         # Display angle
#         angle = frame_data.get("angle", 0)
#         text = "Angle: {:.2f} degrees".format(angle)
#         cv2.putText(frame, text, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        
#         # Display FPS and frame info
#         cv2.putText(frame, f'FPS: {video_fps}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#         cv2.putText(frame, f'Frame: {frame_idx}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
#         # Display path mode
#         path_mode = "Full Path" if show_full_path else f"Last {path_length} frames" if path_length > 0 else "No Path"
#         cv2.putText(frame, f'Path Mode: {path_mode}', (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
#         # Display controls info
#         cv2.putText(frame, "Controls: Space=Pause, P=Toggle Path, +/-=Path Length, Q=Quit", 
#                    (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
#         # Show frame
#         cv2.imshow('Ball Tracking', frame)
        
#         # Handle keyboard input
#         key = cv2.waitKey(frame_delay) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord(' '):  # Space to pause/resume
#             paused = not paused
#             while paused:
#                 key = cv2.waitKey(30) & 0xFF
#                 if key == ord(' '):
#                     paused = not paused
#                 elif key == ord('q'):
#                     break
#         elif key == ord('p'):  # 'p' to toggle path mode
#             if show_full_path:
#                 show_full_path = False
#                 path_length = 30  # Default to last 30 frames
#             elif path_length > 0:
#                 path_length = 0  # Turn off path
#             else:
#                 show_full_path = True  # Back to full path
#         elif key == ord('+') or key == ord('='):  # Increase path length
#             if not show_full_path and path_length > 0:
#                 path_length += 10
#         elif key == ord('-') or key == ord('_'):  # Decrease path length
#             if not show_full_path and path_length > 10:
#                 path_length -= 10
        
#         frame_idx += 1
    
#     cap.release()
#     cv2.destroyAllWindows()

# def save_path_visualization(tracking_data_path, output_path, width=1280, height=720, line_thickness=2):
#     """Create a standalone visualization of the ball's full path and save as an image"""
#     # Load tracking data
#     with open(tracking_data_path, 'r') as f:
#         tracking_data = json.load(f)
    
#     # Create blank image
#     img = np.zeros((height, width, 3), dtype=np.uint8)
#     img.fill(255)  # White background
    
#     # Extract all centroids for visualization
#     all_centroids = tracking_data.get("all_centroids", [])
    
#     if len(all_centroids) <= 1:
#         print("Not enough tracking data for visualization")
#         return
    
#     # Get maximum frame for color gradient calculation
#     max_frame = max([c[0] for c in all_centroids])
    
#     # Draw the path with color gradient
#     for i in range(1, len(all_centroids)):
#         prev_frame, prev_x, prev_y = all_centroids[i-1]
#         curr_frame, curr_x, curr_y = all_centroids[i]
        
#         # Skip if frames are not consecutive (for better visualization)
#         if curr_frame - prev_frame > 5:
#             continue
        
#         # Create color gradient based on frame position
#         ratio = curr_frame / max_frame if max_frame > 0 else 0
#         color = (
#             int(255 * ratio),        # B component
#             0,                        # G component
#             int(255 * (1 - ratio))   # R component
#         )
        
#         cv2.line(img, (prev_x, prev_y), (curr_x, curr_y), color, line_thickness)
    
#     # Mark start and end points
#     first_point = (all_centroids[0][1], all_centroids[0][2])
#     last_point = (all_centroids[-1][1], all_centroids[-1][2])
    
#     cv2.circle(img, first_point, radius=8, color=(0, 255, 0), thickness=-1)  # Green for start
#     cv2.circle(img, last_point, radius=8, color=(0, 0, 255), thickness=-1)   # Red for end
    
#     # Add legend
#     cv2.putText(img, "Start", (first_point[0] + 10, first_point[1]), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#     cv2.putText(img, "End", (last_point[0] + 10, last_point[1]), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
#     # Save image
#     cv2.imwrite(output_path, img)
#     print(f"Path visualization saved to {output_path}")

# if __name__ == "__main__":
#     video_path = os.path.join('videos', '7.mp4')
#     model_path = os.path.join('runs', 'detect', 'train7', 'weights', 'best.pt')
#     tracking_data_path = 'ball_tracking_data.json'
#     visualization_path = 'ball_path_visualization.png'
    
#     # Uncomment the one you want to run:
#     # process_video(video_path, tracking_data_path, model_path)  # First process and save data
#     # replay_with_overlay(video_path, tracking_data_path)  # Replay with path visualization
#     save_path_visualization(tracking_data_path, visualization_path)  # Generate standalone path image


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
        "frames": {},
        "all_centroids": []  # Store all centroids across all frames
    }
    
    frame_idx = 0
    centroid_history = FixedSizeQueue(10)
    
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
                
                # Add to all centroids for complete path
                tracking_data["all_centroids"].append([centroid_x, centroid_y, frame_idx])
        
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
        tracking_data["frames"][frame_idx] = frame_data
        
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
    
    # Extract all centroids for path drawing
    all_centroids = tracking_data["all_centroids"]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get tracking data for this frame
        frame_data = tracking_data["frames"].get(str(frame_idx), {})
        
        # Draw the complete path up to this frame
        centroids_up_to_frame = [c for c in all_centroids if c[2] <= frame_idx]
        if len(centroids_up_to_frame) > 1:
            # Create a polyline of all previous positions
            path_points = [(c[0], c[1]) for c in centroids_up_to_frame]
            path_array = np.array(path_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [path_array], False, (255, 0, 0), 2)
        
        # Draw centroids and bounding boxes
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
        text = "Angle: {:.2f} degrees".format(angle)
        cv2.putText(frame, text, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        
        # Display FPS
        cv2.putText(frame, f'FPS: {video_fps}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Display path length
        if len(centroids_up_to_frame) > 1:
            path_length = calculate_path_length(centroids_up_to_frame)
            cv2.putText(frame, f'Path length: {path_length:.2f} pixels', (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Show frame
        cv2.imshow('Ball Tracking with Path', frame)
        
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

def calculate_path_length(centroids):
    """Calculate the total length of the path in pixels"""
    path_length = 0
    for i in range(1, len(centroids)):
        x1, y1 = centroids[i-1][0], centroids[i-1][1]
        x2, y2 = centroids[i][0], centroids[i][1]
        segment_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        path_length += segment_length
    return path_length

if __name__ == "__main__":
    # Example usage
    # video_path = "path/to/your/video.mp4"
    output_json_path = "tracking_data.json"
    # model_path = "path/to/your/yolo/model.pt"
    video_path = os.path.join('videos', '7.mp4')
    model_path = os.path.join('runs', 'detect', 'train9', 'weights', 'best.pt')
    tracking_data_path = 'ball_tracking_data.json'
    visualization_path = 'ball_path_visualization.png'
    
    # First process the video and save tracking data
    # process_video(video_path, tracking_data_path, model_path)
    
    # Then replay with overlay
    replay_with_overlay(video_path, tracking_data_path)