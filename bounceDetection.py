# import numpy as np
# import cv2
# import math
# import json
# import time
# from ultralytics import YOLO 

# def process_ball_tracking(ball_detections):
#     """
#     Process ball tracking coordinates to find bounce point and filter trajectories
    
#     Args:
#         ball_detections: List of (x, y) coordinates from YOLO
        
#     Returns:
#         bouncing_point: (x, y) coordinates of the bounce
#         corrected_points: List of valid trajectory points
#         rejected_points: List of invalid trajectory points
#     """
#     # Step 1: Find the bouncing point (highest y-value)
#     bouncing_coordinates = (0, 0)
#     bouncing_idx = 0
    
#     for idx, (x, y) in enumerate(ball_detections):
#         if y > bouncing_coordinates[1]:
#             bouncing_coordinates = (x, y)
#             bouncing_idx = idx
            
#     # Step 2: Process pre-bounce trajectory
#     corrected = []
#     for i in range(0, bouncing_idx + 2):
#         if i < len(ball_detections):
#             # Add the point to corrected list - format: (x, y, 0) where 0 is bounce flag
#             # You can extend this format if you have other metrics like radius
#             corrected.append((ball_detections[i][0], ball_detections[i][1], 0))
    
#     # Mark the bouncing point
#     if bouncing_idx < len(corrected):
#         corrected[bouncing_idx] = (corrected[bouncing_idx][0], corrected[bouncing_idx][1], 1)
    
#     # Step 3: Process post-bounce trajectory using angle and distance criteria
#     rejected = []
    
#     if len(ball_detections) >= bouncing_idx + 2:
#         # Get previous point after bounce
#         prev_coord = ball_detections[bouncing_idx + 1]
#         bounce_coord = ball_detections[bouncing_idx]
        
#         # Calculate initial slope after bounce
#         if (prev_coord[0] - bounce_coord[0]) == 0:
#             prev_slope = float('inf')  # Vertical line
#         else:
#             prev_slope = (prev_coord[1] - bounce_coord[1]) / (prev_coord[0] - bounce_coord[0])
        
#         # Process remaining points
#         for i in range(bouncing_idx + 2, len(ball_detections)):
#             current_coord = ball_detections[i]
            
#             # Calculate current slope
#             if (current_coord[0] - prev_coord[0]) == 0:
#                 slope = float('inf')
#             else:
#                 slope = (current_coord[1] - prev_coord[1]) / (current_coord[0] - prev_coord[0])
            
#             # Calculate angle between segments
#             if slope == float('inf') or prev_slope == float('inf'):
#                 angle = 90  # Handle vertical lines
#             else:
#                 tn = abs((slope - prev_slope) / (1.0 + (slope * prev_slope)))
#                 angle = abs(math.atan(tn) * 180.0 / math.pi)
            
#             # Calculate distance between points
#             distance = math.sqrt(((current_coord[1] - prev_coord[1]) ** 2) + 
#                                 ((current_coord[0] - prev_coord[0]) ** 2))
            
#             # Check if point should be kept or rejected
#             # Adjust these thresholds based on your video scale and requirements
#             print(angle, distance,"angle",current_coord,prev_coord)
#             if angle <= 0.0 or distance >= 354:
#                 corrected.append((current_coord[0], current_coord[1], 0))
#                 prev_coord = current_coord
#                 prev_slope = slope
#             else:
#                 rejected.append((current_coord[0], current_coord[1]))
    
#     return bouncing_coordinates, corrected, rejected

# def visualize_trajectory(frame, corrected, rejected, bouncing_point):
#     """
#     Visualize the ball trajectory on the frame
    
#     Args:
#         frame: The video frame to draw on
#         corrected: List of valid trajectory points
#         rejected: List of rejected trajectory points
#         bouncing_point: The bouncing coordinates
    
#     Returns:
#         frame: The frame with visualizations
#     """
#     # Create a copy to avoid modifying the original
#     vis_frame = frame.copy()
    
#     # Draw the bouncing point
#     cv2.circle(vis_frame, (int(bouncing_point[0]), int(bouncing_point[1])), 
#                5, (255, 0, 0), -1)  # Blue circle
    
#     # Draw corrected trajectory points
#     for point in corrected:
#         x, y = int(point[0]), int(point[1])
#         is_bounce = point[2] if len(point) > 2 else 0
        
#         if is_bounce:
#             # Already drew the bounce point above
#             continue
#         else:
#             cv2.circle(vis_frame, (x, y), 3, (0, 255, 0), -1)  # Green circles
    
#     # Draw rejected points
#     for x, y in rejected:
#         cv2.circle(vis_frame, (int(x), int(y)), 3, (0, 0, 255), -1)  # Red circles
    
#     # Draw lines connecting the corrected points to show trajectory
#     for i in range(1, len(corrected)):
#         pt1 = (int(corrected[i-1][0]), int(corrected[i-1][1]))
#         pt2 = (int(corrected[i][0]), int(corrected[i][1]))
#         cv2.line(vis_frame, pt1, pt2, (0, 255, 255), 1)  # Yellow line
        
#     return vis_frame

# def fit_polynomial(corrected_points):
#     """
#     Fit polynomials to the ball trajectory
    
#     Args:
#         corrected_points: List of valid trajectory points [(x, y, flag), ...]
        
#     Returns:
#         pre_bounce_fit: Function for pre-bounce trajectory
#         post_bounce_fit: Function for post-bounce trajectory
#     """
#     # Extract x and y coordinates
#     x_coords = np.array([p[0] for p in corrected_points])
#     y_coords = np.array([p[1] for p in corrected_points])
#     bounce_flags = np.array([p[2] for p in corrected_points])
    
#     # Find bounce index
#     bounce_idx = np.where(bounce_flags == 1)[0]
#     if len(bounce_idx) == 0:
#         return None, None
#     bounce_idx = bounce_idx[0]
    
#     # Split into pre-bounce and post-bounce
#     pre_x = x_coords[:bounce_idx+1]
#     pre_y = y_coords[:bounce_idx+1]
#     post_x = x_coords[bounce_idx:]
#     post_y = y_coords[bounce_idx:]
    
#     # Fit quadratic polynomials
#     if len(pre_x) >= 3:
#         pre_coeffs = np.polyfit(pre_x, pre_y, 2)
#         pre_bounce_fit = np.poly1d(pre_coeffs)
#     else:
#         pre_bounce_fit = None
        
#     if len(post_x) >= 3:
#         post_coeffs = np.polyfit(post_x, post_y, 2)
#         post_bounce_fit = np.poly1d(post_coeffs)
#     else:
#         post_bounce_fit = None
        
#     return pre_bounce_fit, post_bounce_fit



# def predict_cricket_video(video_path, model_path):
#     """Main function to predict ball trajectory in cricket video"""
#     # Initialize variables
#     paused = False
#     impact_detected = False
#     impact_handled = False
#     future_trajectory = None
    
#     # Load YOLO model
#     model = YOLO(model_path)
    
#     # Open video
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error: Could not open video {video_path}")
#         return
    
#     # Get video properties
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
#     # Create writer for output video
#     output_path = 'output_prediction.mp4'
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
#     # Initialize tracking data structure
#     tracking_data = {
#         "video_info": {
#             "fps": fps,
#             "frame_count": frame_count,
#             "width": width,
#             "height": height
#         },
#         "frames": []
#     }
    
#     # Calculate frame delay for display
#     frame_delay = int(1000 / fps) if fps > 0 else 30
    
#     # Process video frames
#     current_frame_idx = 0
#     ball_detections = []
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Create a copy for display
#         display_frame = frame.copy()
        
#         # Run YOLOv8 detection
#         results = model(frame)
        
#         # Initialize frame data
#         frame_data = {
#             "frame_number": current_frame_idx,
#             "objects": {},
#             "distances": {}
#         }
        
        
#         # Process detection results
#         for r in results:
#             boxes = r.boxes
#             for box in boxes:
#                 # Get bounding box coordinates
#                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
#                 # Get class information
#                 class_id = int(box.cls[0])
#                 class_name = model.names[class_id]
#                 confidence = float(box.conf[0])
                
#                 # Calculate center point
#                 center_x = (x1 + x2) / 2
#                 center_y = (y1 + y2) / 2
                
#                 # Store object information
#                 if class_name not in frame_data["objects"]:
#                     frame_data["objects"][class_name] = []

#                 if(class_name == 'ball'):
#                     ball_detections.append((center_x, center_y))
                
#                 frame_data["objects"][class_name].append({
#                     "bbox": [x1, y1, x2, y2],
#                     "center": [center_x, center_y],
#                     "confidence": confidence
#                 })
                
#                 # Draw bounding box
#                 cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(display_frame, f"{class_name} {confidence:.2f}", 
#                             (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
#         # Calculate distances between objects
#         # calculate_distances(frame_data)
        
#         # Store frame data
#         tracking_data["frames"].append(frame_data)
        
       
#         cv2.putText(display_frame, f"Frame: {current_frame_idx}", 
#                     (width - 150, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
#                     0.6, (255, 255, 255), 2)
        
#         # Show frame
#         cv2.imshow("Cricket Ball Trajectory Analysis", display_frame)
        
#         # Write frame to output video
#         out.write(display_frame)
        
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
        
#         current_frame_idx += 1
    
#     # Release resources
#     print("Releasing resources...",ball_detections)
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
    
#     # Save tracking data to file
#     with open('ball_tracking_data.json', 'w') as f:
#         json.dump(tracking_data, f, indent=2)
    
#     print(f"Analysis complete. Output video saved to {output_path}")
#     print(f"Tracking data saved to ball_tracking_data.json")
    
#     return ball_detections




# def example_usage():
#     """
#     Example of how to use the functions
#     """
#     # Mock YOLO detections - replace with your actual data
#     # ball_detections = [
#     #     (100, 100), (110, 130), (120, 170), (130, 220), (140, 280),  # Pre-bounce
#     #     (150, 350),  # Bounce point
#     #     (160, 320), (170, 300), (180, 290),  # Post-bounce
#     #     (500, 100)   # Outlier that should be rejected
#     # ]
#     video_path = os.path.join('videos', 'ball1.mov')
#     model_path = os.path.join('runs', 'detect', 'train11', 'best.pt')
#     ball_detections=predict_cricket_video(video_path, model_path)
    
#     # Process the detections
#     bouncing_point, corrected, rejected = process_ball_tracking(ball_detections)
    
#     print(f"Bouncing point: {bouncing_point}")
#     print(f"Corrected points: {corrected}")
#     print(f"Rejected points: {rejected}")
    
#     # Create a blank frame for visualization
#     frame = np.zeros((600, 800, 3), dtype=np.uint8)
#     frame.fill(255)  # White background
    
#     # Visualize trajectory
#     vis_frame = visualize_trajectory(frame, corrected, rejected, bouncing_point)
    
#     # Fit polynomials
#     pre_bounce_fit, post_bounce_fit = fit_polynomial(corrected)
    
#     # Draw the fitted curves
#     if pre_bounce_fit is not None:
#         x_range = np.linspace(corrected[0][0], bouncing_point[0], 100)
#         for i in range(len(x_range)-1):
#             pt1 = (int(x_range[i]), int(pre_bounce_fit(x_range[i])))
#             pt2 = (int(x_range[i+1]), int(pre_bounce_fit(x_range[i+1])))
#             cv2.line(vis_frame, pt1, pt2, (255, 0, 255), 2)  # Magenta line
    
#     if post_bounce_fit is not None:
#         x_range = np.linspace(bouncing_point[0], corrected[-1][0], 100)
#         for i in range(len(x_range)-1):
#             pt1 = (int(x_range[i]), int(post_bounce_fit(x_range[i])))
#             pt2 = (int(x_range[i+1]), int(post_bounce_fit(x_range[i+1])))
#             cv2.line(vis_frame, pt1, pt2, (255, 165, 0), 2)  # Orange line
    
#     # Display results
#     cv2.imshow("Ball Trajectory", vis_frame)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # if __name__ == "__main__":

# #     example_usage()


# if __name__ == "__main__":
#     import os
    
#     # Define paths for video and model
    
#     example_usage()
    
#     # Execute the main function
#     # ball_detctions=predict_cricket_video(video_path, model_path)
#     # bouncing_point, corrected, rejected = process_ball_tracking(ball_detctions)
    
#     # print(f"Bouncing point: {bouncing_point}")
#     # print(f"Corrected points: {corrected}")
#     # print(f"Rejected points: {rejected}")






import numpy as np
import cv2
import math
import json
import time
import os
from ultralytics import YOLO 

def process_ball_tracking(ball_detections):
    """
    Process ball tracking coordinates to find bounce point and filter trajectories
    
    Args:
        ball_detections: List of (x, y) coordinates from YOLO
        
    Returns:
        bouncing_point: (x, y) coordinates of the bounce
        corrected_points: List of valid trajectory points
        rejected_points: List of invalid trajectory points
    """
    # Step 1: Find the bouncing point (highest y-value)
    bouncing_coordinates = (0, 0)
    bouncing_idx = 0
    
    for idx, (x, y) in enumerate(ball_detections):
        if y > bouncing_coordinates[1]:
            bouncing_coordinates = (x, y)
            bouncing_idx = idx
            
    # Step 2: Process pre-bounce trajectory
    corrected = []
    for i in range(0, bouncing_idx + 2):
        if i < len(ball_detections):
            # Add the point to corrected list - format: (x, y, 0) where 0 is bounce flag
            # You can extend this format if you have other metrics like radius
            corrected.append((ball_detections[i][0], ball_detections[i][1], 0))
    
    # Mark the bouncing point
    if bouncing_idx < len(corrected):
        corrected[bouncing_idx] = (corrected[bouncing_idx][0], corrected[bouncing_idx][1], 1)
    
    # Step 3: Process post-bounce trajectory using angle and distance criteria

    rejected = []
    
    if len(ball_detections) >= bouncing_idx + 2:
        # Get previous point after bounce
        prev_coord = ball_detections[bouncing_idx + 1]
        bounce_coord = ball_detections[bouncing_idx]
        
        # Calculate initial slope after bounce
        if (prev_coord[0] - bounce_coord[0]) == 0:
            prev_slope = float('inf')  # Vertical line
        else:
            prev_slope = (prev_coord[1] - bounce_coord[1]) / (prev_coord[0] - bounce_coord[0])
        
        # Process remaining points
        for i in range(bouncing_idx + 2, len(ball_detections)):
            current_coord = ball_detections[i]
            
            # Calculate current slope
            if (current_coord[0] - prev_coord[0]) == 0:
                slope = float('inf')
            else:
                slope = (current_coord[1] - prev_coord[1]) / (current_coord[0] - prev_coord[0])
            
            # Calculate angle between segments
            if slope == float('inf') or prev_slope == float('inf'):
                angle = 90  # Handle vertical lines
            else:
                tn = abs((slope - prev_slope) / (1.0 + (slope * prev_slope)))
                angle = abs(math.atan(tn) * 180.0 / math.pi)
            
            # Calculate distance between points
            distance = math.sqrt(((current_coord[1] - prev_coord[1]) ** 2) + 
                                ((current_coord[0] - prev_coord[0]) ** 2))
            
            # Check if point should be kept or rejected
            # Adjust these thresholds based on your video scale and requirements
            print(angle, distance,"angle",current_coord,prev_coord)
            if angle <= 35 or distance >= 500:
                corrected.append((current_coord[0], current_coord[1], 0))
                prev_coord = current_coord
                prev_slope = slope
            else:
                rejected.append((current_coord[0], current_coord[1]))
    
    return bouncing_coordinates, corrected, rejected

def visualize_trajectory(frame, corrected, rejected, bouncing_point, pre_bounce_fit=None, post_bounce_fit=None):
    """
    Visualize the ball trajectory on the frame
    
    Args:
        frame: The video frame to draw on
        corrected: List of valid trajectory points
        rejected: List of rejected trajectory points
        bouncing_point: The bouncing coordinates
        pre_bounce_fit: Polynomial function for pre-bounce path
        post_bounce_fit: Polynomial function for post-bounce path
    
    Returns:
        frame: The frame with visualizations
    """
    # Create a copy to avoid modifying the original
    vis_frame = frame.copy()
    
    # Draw the bouncing point
    cv2.circle(vis_frame, (int(bouncing_point[0]), int(bouncing_point[1])), 
               5, (255, 0, 0), -1)  # Blue circle
    
    # Draw corrected trajectory points
    # for point in corrected:
    #     x, y = int(point[0]), int(point[1])
    #     is_bounce = point[2] if len(point) > 2 else 0
        
    #     if is_bounce:
    #         # Already drew the bounce point above
    #         continue
    #     else:
    #         cv2.circle(vis_frame, (x, y), 3, (0, 255, 0), -1)  # Green circles
    
    # Draw rejected points
    for x, y in rejected:
        cv2.circle(vis_frame, (int(x), int(y)), 3, (0, 0, 255), -1)  # Red circles
    
    # Draw lines connecting the corrected points to show trajectory
    for i in range(1, len(corrected)):
        pt1 = (int(corrected[i-1][0]), int(corrected[i-1][1]))
        pt2 = (int(corrected[i][0]), int(corrected[i][1]))
        cv2.line(vis_frame, pt1, pt2, (0, 255, 255), 1)  # Yellow line
        
    # Draw fitted curves if available
    if pre_bounce_fit is not None:
        # Find the starting x for pre-bounce fit (first corrected point)
        start_x = corrected[0][0] if len(corrected) > 0 else 0
        # Find the ending x for pre-bounce fit (bounce point)
        end_x = bouncing_point[0]
        
        # Draw the pre-bounce curve
        x_range = np.linspace(start_x, end_x, 100)
        for i in range(len(x_range)-1):
            pt1 = (int(x_range[i]), int(pre_bounce_fit(x_range[i])))
            pt2 = (int(x_range[i+1]), int(pre_bounce_fit(x_range[i+1])))
            cv2.line(vis_frame, pt1, pt2, (255, 0, 255), 2)  # Magenta line
    
    # if post_bounce_fit is not None:
    #     # Find the starting x for post-bounce fit (bounce point)
    #     start_x = bouncing_point[0]
    #     # Find the ending x for post-bounce fit (last corrected point)
    #     end_x = corrected[-1][0] if len(corrected) > 0 else start_x
        
    # #     # Draw the post-bounce curve
    #     x_range = np.linspace(start_x, end_x, 100)
    #     for i in range(len(x_range)-1):
    #         pt1 = (int(x_range[i]), int(post_bounce_fit(x_range[i])))
    #         pt2 = (int(x_range[i+1]), int(post_bounce_fit(x_range[i+1])))
    #         cv2.line(vis_frame, pt1, pt2, (255, 165, 0), 2)  # Orange line
            
    return vis_frame

def fit_polynomial(corrected_points):
    """
    Fit polynomials to the ball trajectory
    
    Args:
        corrected_points: List of valid trajectory points [(x, y, flag), ...]
        
    Returns:
        pre_bounce_fit: Function for pre-bounce trajectory
        post_bounce_fit: Function for post-bounce trajectory
    """
    # Extract x and y coordinates
    x_coords = np.array([p[0] for p in corrected_points])
    y_coords = np.array([p[1] for p in corrected_points])
    bounce_flags = np.array([p[2] for p in corrected_points])
    
    # Find bounce index
    bounce_idx = np.where(bounce_flags == 1)[0]
    if len(bounce_idx) == 0:
        return None, None
    bounce_idx = bounce_idx[0]
    
    # Split into pre-bounce and post-bounce
    pre_x = x_coords[:bounce_idx+1]
    pre_y = y_coords[:bounce_idx+1]
    post_x = x_coords[bounce_idx:]
    post_y = y_coords[bounce_idx:]
    
    # Fit quadratic polynomials
    if len(pre_x) >= 3:
        pre_coeffs = np.polyfit(pre_x, pre_y, 2)
        pre_bounce_fit = np.poly1d(pre_coeffs)
    else:
        pre_bounce_fit = None
        
    if len(post_x) >= 3:
        post_coeffs = np.polyfit(post_x, post_y, 2)
        post_bounce_fit = np.poly1d(post_coeffs)
    else:
        post_bounce_fit = None
        
    return pre_bounce_fit, post_bounce_fit

def predict_cricket_video(video_path, model_path):
    """Main function to predict ball trajectory in cricket video"""
    # Initialize variables
    paused = False
    impact_detected = False
    impact_handled = False
    
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
    output_path = 'output_with_trajectory.mp4'
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
    
    # First pass: collect all ball detections
    print("First pass: Collecting ball detections...")
    ball_detections = []
    current_frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLOv8 detection
        results = model(frame)
        
        # Initialize frame data
        frame_data = {
            "frame_number": current_frame_idx,
            "objects": {}
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

                if class_name == 'ball':
                    ball_detections.append((center_x, center_y))
                
                frame_data["objects"][class_name].append({
                    "bbox": [x1, y1, x2, y2],
                    "center": [center_x, center_y],
                    "confidence": confidence
                })
        
        # Store frame data
        tracking_data["frames"].append(frame_data)
        current_frame_idx += 1
    
    # Process the ball trajectory
    print(f"Processing trajectory from {len(ball_detections)} ball detections...")
    bouncing_point, corrected, rejected = process_ball_tracking(ball_detections)
    
    # Fit polynomials to the trajectory
    pre_bounce_fit, post_bounce_fit = fit_polynomial(corrected)
    
    # Second pass: Draw trajectories on video
    print("Second pass: Drawing trajectories on video...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to beginning
    current_frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create a copy for display and drawing
        display_frame = frame.copy()
        
        # Run YOLOv8 detection for this frame (for drawing current detections)
        results = model(frame)
        
        # Draw current detections
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
                
                # Draw bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"{class_name} {confidence:.2f}", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw the complete trajectory on every frame
        vis_frame = visualize_trajectory(display_frame, corrected, rejected, bouncing_point, 
                                        pre_bounce_fit, post_bounce_fit)
        
        # Add frame counter
        cv2.putText(vis_frame, f"Frame: {current_frame_idx}", 
                    (width - 150, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 2)
        
        # Add legend at the top right corner
        legend_y = 30
        cv2.putText(vis_frame, "Legend:", (width - 150, legend_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        legend_y += 20
        cv2.putText(vis_frame, "Ball Detection", (width - 150, legend_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.circle(vis_frame, (width - 160, legend_y - 5), 3, (0, 255, 0), -1)
        legend_y += 20
        cv2.putText(vis_frame, "Bounce Point", (width - 150, legend_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.circle(vis_frame, (width - 160, legend_y - 5), 5, (255, 0, 0), -1)
        legend_y += 20
        cv2.putText(vis_frame, "Rejected Point", (width - 150, legend_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.circle(vis_frame, (width - 160, legend_y - 5), 3, (0, 0, 255), -1)
        legend_y += 20
        cv2.putText(vis_frame, "Pre-bounce Fit", (width - 150, legend_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        cv2.line(vis_frame, (width - 180, legend_y - 5), (width - 160, legend_y - 5), (255, 0, 255), 2)
        legend_y += 20
        cv2.putText(vis_frame, "Post-bounce Fit", (width - 150, legend_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
        cv2.line(vis_frame, (width - 180, legend_y - 5), (width - 160, legend_y - 5), (255, 165, 0), 2)
        
        # Show frame
        cv2.imshow("Cricket Ball Trajectory Analysis", vis_frame)
        
        # Write frame to output video
        out.write(vis_frame)
        
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
    print(f"Detected {len(ball_detections)} ball positions")
    print(f"Bouncing point at {bouncing_point}")
    print(f"Maintained {len(corrected)} trajectory points")
    print(f"Rejected {len(rejected)} outlier points")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Save tracking data to file
    with open('ball_tracking_data.json', 'w') as f:
        json.dump(tracking_data, f, indent=2)
    
    print(f"Analysis complete. Output video saved to {output_path}")
    print(f"Tracking data saved to ball_tracking_data.json")
    
    return ball_detections, bouncing_point, corrected, rejected

def example_usage():
    """
    Example of how to use the functions
    """
    # Define paths for video and model
    video_path = os.path.join('videos', 'ball4.mp4')
    model_path = os.path.join('runs', 'detect', 'train11', 'best.pt')
    
    # Execute main function
    ball_detections, bouncing_point, corrected, rejected = predict_cricket_video(video_path, model_path)
    
    print(f"Bouncing point: {bouncing_point}")
    print(f"Corrected points: {len(corrected)}")
    print(f"Rejected points: {len(rejected)}")

if __name__ == "__main__":
    example_usage()