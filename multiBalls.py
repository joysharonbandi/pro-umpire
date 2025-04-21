# from collections import deque
# from ultralytics import YOLO
# import math
# import time
# import cv2
# import os
# import json
# import numpy as np
# import threading
# from flask import Flask, Response, render_template, request, jsonify

# # Constants
# INACTIVITY_THRESHOLD = 30  # Frames of inactivity before considering a new bowling action
# ACTIVITY_REGION_Y = 0.2    # Top 20% of the frame is considered the bowling area
# NEW_BOWL_DISTANCE = 50     # Minimum distance for a new detection to be considered a new bowl

# def angle_between_lines(m1, m2=1):
#     if m1 != -1/m2:
#         angle = math.degrees(math.atan(abs((m2 - m1) / (1 + m1 * m2))))
#         return angle
#     else:
#         return 90.0

# class BallTracker:
#     def __init__(self, max_history=10):
#         self.centroid_history = deque(maxlen=max_history)
#         self.last_seen = 0  # Frame counter for when this ball was last seen
#         self.active = True  # Whether this ball is currently active
    
#     def add_centroid(self, centroid):
#         self.centroid_history.append(centroid)
    
#     def get_history(self):
#         return list(self.centroid_history)
    
#     def calculate_trajectory(self):
#         """Calculate angle and future positions based on centroid history"""
#         angle = 0
#         future_positions = []
        
#         if len(self.centroid_history) > 1:
#             centroid_list = list(self.centroid_history)
#             x_diff = centroid_list[-1][0] - centroid_list[-2][0]
#             y_diff = centroid_list[-1][1] - centroid_list[-2][1]
            
#             if x_diff != 0:
#                 m1 = y_diff/x_diff
#                 if m1 == 1:
#                     angle = 90
#                 elif m1 != 0:
#                     angle = 90-angle_between_lines(m1)
            
#             # Calculate 5 future positions
#             future_positions = [centroid_list[-1]]
#             for i in range(1, 5):
#                 future_pos = (
#                     centroid_list[-1][0] + x_diff * i,
#                     centroid_list[-1][1] + y_diff * i
#                 )
#                 future_positions.append(future_pos)
        
#         return angle, future_positions
    
#     def __len__(self):
#         return len(self.centroid_history)

# class BallTrackingSystem:
#     def __init__(self, model_path):
#         self.model = YOLO(model_path)
#         self.cap = None
#         self.video_path = None
#         self.frame_count = 0
#         self.fps = 0
#         self.ball_trackers = {}
#         self.tracking_data = {}
#         self.max_frames_missing = 5
#         self.frame_idx = 0
#         self.paused = False
#         self.running = False
#         self.current_frame = None
#         self.frame_height = 0
#         self.frame_width = 0
        
#         # Bowling detection variables
#         self.last_activity_frame = 0
#         self.active_ball_id = None
#         self.new_bowl_detected = False
        
#         # Overlay settings
#         self.show_bounding_boxes = True
#         self.show_centroids = True
#         self.show_trajectories = True
#         self.show_angles = True
#         self.show_ball_ids = True
#         self.show_stats = True
#         self.clear_paths_on_new_bowl = True  # Default to clearing paths
        
#         # Colors for different balls
#         self.colors = [
#             (0, 0, 255),     # Red
#             (0, 255, 0),     # Green
#             (255, 0, 0),     # Blue
#             (0, 255, 255),   # Yellow
#             (255, 0, 255),   # Magenta
#             (255, 255, 0),   # Cyan
#             (128, 0, 0),     # Dark blue
#             (0, 128, 0),     # Dark green
#             (0, 0, 128),     # Dark red
#             (128, 128, 0)    # Dark cyan
#         ]
    
#     def open_video(self, video_path):
#         self.video_path = video_path
#         self.cap = cv2.VideoCapture(video_path)
#         self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         self.fps = self.cap.get(cv2.CAP_PROP_FPS)
#         self.frame_idx = 0
#         self.ball_trackers = {}
#         self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
#         self.tracking_data = {
#             "video_info": {
#                 "fps": self.fps,
#                 "frame_count": self.frame_count,
#                 "width": self.frame_width,
#                 "height": self.frame_height
#             },
#             "frames": {},
#             "ball_count": 0,
#             "bowl_count": 0
#         }
        
#         return {
#             "fps": self.fps,
#             "frame_count": self.frame_count,
#             "width": self.frame_width,
#             "height": self.frame_height
#         }
    
#     def detect_new_bowl(self, ball_id, centroid_y):
#         """Detect if a new ball has been bowled based on position and timing"""
#         # Check if the ball appears in the bowling area (top of the frame)
#         is_in_bowling_area = centroid_y < (self.frame_height * ACTIVITY_REGION_Y)
        
#         # Condition 1: Ball appears in bowling area after period of inactivity
#         if is_in_bowling_area and (self.frame_idx - self.last_activity_frame) > INACTIVITY_THRESHOLD:
#             self.last_activity_frame = self.frame_idx
#             self.active_ball_id = ball_id
#             self.tracking_data["bowl_count"] += 1
#             return True
        
#         # Condition 2: New ball appears in bowling area while another is still active
#         if is_in_bowling_area and self.active_ball_id is not None and ball_id != self.active_ball_id:
#             # Check if this is truly a new ball and not just a tracking ID switch
#             if ball_id not in self.ball_trackers:
#                 self.last_activity_frame = self.frame_idx
#                 self.active_ball_id = ball_id
#                 self.tracking_data["bowl_count"] += 1
#                 return True
        
#         return False
    
#     def process_frame(self):
#         if self.cap is None or not self.cap.isOpened():
#             return None
        
#         ret, frame = self.cap.read()
#         if not ret:
#             self.running = False
#             return None
        
#         # Process frame with YOLO
#         results = self.model.track(frame, persist=True, conf=0.35, verbose=False)
        
#         # Data for this frame
#         frame_data = {
#             "balls": {},
#             "new_bowl_detected": False
#         }
        
#         if results[0].boxes.id is not None:
#             # Extract IDs, bounding boxes, and calculate centroids
#             ids = results[0].boxes.id.cpu().numpy().astype(int)
#             boxes = results[0].boxes.xyxy.cpu().numpy()
            
#             # Update all active trackers with "not seen in this frame"
#             active_ids = set()
#             new_bowl_detected = False
            
#             for i, box_id in enumerate(ids):
#                 ball_id = int(box_id)
#                 active_ids.add(ball_id)
                
#                 x1, y1, x2, y2 = boxes[i]
#                 centroid_x = int((x1+x2)/2)
#                 centroid_y = int((y1+y2)/2)
                
#                 # Check if this is a new bowl
#                 if ball_id not in self.ball_trackers:
#                     is_new_bowl = self.detect_new_bowl(ball_id, centroid_y)
#                     if is_new_bowl and self.clear_paths_on_new_bowl:
#                         # Deactivate all previous balls when a new bowl is detected
#                         for tracker_id in self.ball_trackers:
#                             self.ball_trackers[tracker_id].active = False
#                         new_bowl_detected = True
                
#                 # Create new tracker if this is a new ball
#                 if ball_id not in self.ball_trackers:
#                     self.ball_trackers[ball_id] = BallTracker()
#                     self.tracking_data["ball_count"] += 1
                
#                 # Add centroid to history
#                 self.ball_trackers[ball_id].add_centroid((centroid_x, centroid_y))
#                 self.ball_trackers[ball_id].last_seen = self.frame_idx
#                 self.ball_trackers[ball_id].active = True  # Ensure it's marked as active
                
#                 # Calculate trajectory
#                 angle, future_positions = self.ball_trackers[ball_id].calculate_trajectory()
                
#                 # Store data for this ball
#                 frame_data["balls"][ball_id] = {
#                     "centroid": [centroid_x, centroid_y],
#                     "bbox": [int(x1), int(y1), int(x2), int(y2)],
#                     "angle": angle,
#                     "future_positions": [[int(pos[0]), int(pos[1])] for pos in future_positions],
#                     "active": True
#                 }
            
#             # If we detected a new bowl, store that in the frame data
#             frame_data["new_bowl_detected"] = new_bowl_detected
            
#             # Remove trackers for balls that haven't been seen for too long
#             inactive_ids = []
#             for ball_id, tracker in self.ball_trackers.items():
#                 if ball_id not in active_ids and (self.frame_idx - tracker.last_seen) > self.max_frames_missing:
#                     inactive_ids.append(ball_id)
            
#             for ball_id in inactive_ids:
#                 del self.ball_trackers[ball_id]
        
#         # Store frame data
#         self.tracking_data["frames"][self.frame_idx] = frame_data
        
#         # Apply overlays if enabled
#         frame_with_overlay = self.apply_overlays(frame.copy(), frame_data)
        
#         self.current_frame = frame_with_overlay  # Keep current frame for streaming
#         self.frame_idx += 1
        
#         if self.frame_idx >= self.frame_count:
#             self.running = False
#             self.cap.release()
#             self.cap = None
        
#         return frame_with_overlay
    
#     def apply_overlays(self, frame, frame_data):
#         """Apply selected overlays to the frame"""
#         balls_data = frame_data.get("balls", {})
        
#         # Draw bowling area indicator line
#         if self.show_stats:
#             bowling_line_y = int(self.frame_height * ACTIVITY_REGION_Y)
#             cv2.line(frame, (0, bowling_line_y), (self.frame_width, bowling_line_y), 
#                    (255, 255, 255), 1, cv2.LINE_AA)  # Changed LINE_DASHED to LINE_AA
            
#             # Label the bowling area
#             cv2.putText(frame, "Bowling Area", (10, bowling_line_y - 10), 
#                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
#         # Process each detected ball
#         for ball_id, ball_data in balls_data.items():
#             # Use modulo to cycle through colors if we have more balls than colors
#             color_idx = int(ball_id) % len(self.colors)
#             color = self.colors[color_idx]
            
#             # Check if the ball is active in the tracker
#             is_active = True
#             if ball_id in self.ball_trackers:
#                 is_active = self.ball_trackers[ball_id].active
            
#             # Draw centroid
#             if self.show_centroids:
#                 centroid = ball_data.get("centroid", [])
#                 if centroid:
#                     cv2.circle(frame, (centroid[0], centroid[1]), radius=3, color=color, thickness=-1)
            
#             # Draw bounding box
#             if self.show_bounding_boxes:
#                 bbox = ball_data.get("bbox", [])
#                 if bbox:
#                     cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    
#                     # Put ball ID on the bbox
#                     if self.show_ball_ids:
#                         status = "Active" if is_active else "Inactive"
#                         cv2.putText(frame, f"Ball {ball_id} ({status})", 
#                                   (bbox[0], bbox[1] - 10), 
#                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
#             # Draw future trajectory only for active balls
#             if self.show_trajectories and is_active:
#                 future_positions = ball_data.get("future_positions", [])
#                 if len(future_positions) > 1:
#                     for i in range(1, len(future_positions)):
#                         cv2.line(frame, 
#                                (future_positions[i-1][0], future_positions[i-1][1]), 
#                                (future_positions[i][0], future_positions[i][1]), 
#                                color, 2)
#                         cv2.circle(frame, 
#                                  (future_positions[i][0], future_positions[i][1]), 
#                                  radius=3, color=color, thickness=-1)
            
#             # Display angle only for active balls
#             if self.show_angles and is_active:
#                 angle = ball_data.get("angle", 0)
#                 centroid = ball_data.get("centroid", [])
#                 if centroid:
#                     cv2.putText(frame, f"{angle:.1f}°", 
#                               (centroid[0] + 10, centroid[1] + 10), 
#                               cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
        
#         # Draw stats
#         if self.show_stats:
#             # Display ball count
#             cv2.putText(frame, f'Total balls: {self.tracking_data["ball_count"]}', 
#                       (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
#             # Display bowl count
#             cv2.putText(frame, f'Bowl count: {self.tracking_data["bowl_count"]}', 
#                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
#             # Display active balls in this frame
#             cv2.putText(frame, f'Active balls: {len(balls_data)}', 
#                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
#             # Display frame number
#             cv2.putText(frame, f'Frame: {self.frame_idx}/{self.frame_count}', 
#                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
#         return frame
    
#     def start_processing(self):
#         """Start processing video frames in a loop"""
#         self.running = True
#         while self.running:
#             if not self.paused:
#                 self.process_frame()
#             time.sleep(1/self.fps)  # Sleep to maintain video FPS
    
#     def stop_processing(self):
#         """Stop the processing loop"""
#         self.running = False
#         if self.cap is not None:
#             self.cap.release()
#             self.cap = None
    
#     def toggle_pause(self):
#         """Toggle pause state"""
#         self.paused = not self.paused
#         return self.paused
    
#     def save_tracking_data(self, output_path):
#         """Save tracking data to a JSON file"""
#         with open(output_path, 'w') as f:
#             json.dump(self.tracking_data, f)
#         return True
    
#     def toggle_clear_paths(self):
#         """Toggle whether to clear paths on new bowls"""
#         self.clear_paths_on_new_bowl = not self.clear_paths_on_new_bowl
#         return self.clear_paths_on_new_bowl
    
#     def get_frame_as_jpg(self):
#         """Convert current frame to JPEG bytes"""
#         if self.current_frame is None:
#             return None
#         ret, jpeg = cv2.imencode('.jpg', self.current_frame)
#         return jpeg.tobytes()

# # Initialize Flask app
# app = Flask(__name__)

# # Initialize tracking system
# model_path = os.path.join('runs', 'detect', 'train9', 'weights', 'best.pt')
# tracking_system = BallTrackingSystem(model_path)
# processing_thread = None

# # Generate HTML template
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     """Video streaming route"""
#     def generate():
#         while tracking_system.running:
#             frame_jpg = tracking_system.get_frame_as_jpg()
#             if frame_jpg is not None:
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + frame_jpg + b'\r\n\r\n')
#             time.sleep(0.01)  # Small delay
    
#     return Response(generate(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/open_video', methods=['POST'])
# def open_video():
#     """Open a video file"""
#     data = request.get_json()
#     video_path = data.get('video_path')
    
#     if not os.path.exists(video_path):
#         return jsonify({'success': False, 'error': 'Video file not found'})
    
#     # Stop any existing processing
#     global processing_thread
#     if processing_thread and processing_thread.is_alive():
#         tracking_system.stop_processing()
#         processing_thread.join()
    
#     # Open the video
#     video_info = tracking_system.open_video(video_path)
    
#     # Start processing in a new thread
#     processing_thread = threading.Thread(target=tracking_system.start_processing)
#     processing_thread.daemon = True
#     processing_thread.start()
    
#     return jsonify({'success': True, 'video_info': video_info})

# @app.route('/toggle_overlay', methods=['POST'])
# def toggle_overlay():
#     """Toggle overlay settings"""
#     data = request.get_json()
#     setting = data.get('setting')
#     value = data.get('value', None)
    
#     if setting == 'bounding_boxes':
#         tracking_system.show_bounding_boxes = value if value is not None else not tracking_system.show_bounding_boxes
#         return jsonify({'success': True, 'value': tracking_system.show_bounding_boxes})
#     elif setting == 'centroids':
#         tracking_system.show_centroids = value if value is not None else not tracking_system.show_centroids
#         return jsonify({'success': True, 'value': tracking_system.show_centroids})
#     elif setting == 'trajectories':
#         tracking_system.show_trajectories = value if value is not None else not tracking_system.show_trajectories
#         return jsonify({'success': True, 'value': tracking_system.show_trajectories})
#     elif setting == 'angles':
#         tracking_system.show_angles = value if value is not None else not tracking_system.show_angles
#         return jsonify({'success': True, 'value': tracking_system.show_angles})
#     elif setting == 'ball_ids':
#         tracking_system.show_ball_ids = value if value is not None else not tracking_system.show_ball_ids
#         return jsonify({'success': True, 'value': tracking_system.show_ball_ids})
#     elif setting == 'stats':
#         tracking_system.show_stats = value if value is not None else not tracking_system.show_stats
#         return jsonify({'success': True, 'value': tracking_system.show_stats})
#     elif setting == 'clear_paths':
#         tracking_system.clear_paths_on_new_bowl = value if value is not None else not tracking_system.clear_paths_on_new_bowl
#         return jsonify({'success': True, 'value': tracking_system.clear_paths_on_new_bowl})
#     else:
#         return jsonify({'success': False, 'error': 'Invalid setting'})

# @app.route('/toggle_pause', methods=['POST'])
# def toggle_pause():
#     """Toggle pause/play"""
#     paused = tracking_system.toggle_pause()
#     return jsonify({'success': True, 'paused': paused})

# @app.route('/save_data', methods=['POST'])
# def save_data():
#     """Save tracking data to file"""
#     data = request.get_json()
#     output_path = data.get('output_path', 'tracking_data.json')
    
#     success = tracking_system.save_tracking_data(output_path)
#     return jsonify({'success': success})

# @app.route('/stop_video', methods=['POST'])
# def stop_video():
#     """Stop video processing"""
#     global processing_thread
#     if processing_thread and processing_thread.is_alive():
#         tracking_system.stop_processing()
#         processing_thread.join()
    
#     return jsonify({'success': True})

# # Create the HTML template
# def create_templates():
#     """Create the HTML template directory and file"""
#     if not os.path.exists('templates'):
#         os.makedirs('templates')
        
#     with open('templates/index.html', 'w') as f:
#         f.write('''
# <!DOCTYPE html>
# <html>
# <head>
#     <title>Ball Tracking System</title>
#     <style>
#         body {
#             font-family: Arial, sans-serif;
#             margin: 0;
#             padding: 20px;
#             background-color: #f4f4f4;
#         }
#         .container {
#             max-width: 1200px;
#             margin: 0 auto;
#         }
#         .video-container {
#             margin-top: 20px;
#             text-align: center;
#         }
#         #video-feed {
#             max-width: 100%;
#             border: 1px solid #ddd;
#         }
#         .controls {
#             margin-top: 20px;
#             padding: 15px;
#             background-color: #fff;
#             border-radius: 5px;
#             box-shadow: 0 2px 5px rgba(0,0,0,0.1);
#         }
#         .control-group {
#             margin-bottom: 15px;
#         }
#         h2 {
#             margin-top: 0;
#             color: #333;
#         }
#         button {
#             background-color: #4CAF50;
#             color: white;
#             border: none;
#             padding: 8px 16px;
#             text-align: center;
#             text-decoration: none;
#             display: inline-block;
#             font-size: 14px;
#             margin: 4px 2px;
#             cursor: pointer;
#             border-radius: 4px;
#         }
#         button:hover {
#             background-color: #45a049;
#         }
#         input[type="text"] {
#             padding: 8px;
#             width: 300px;
#             border: 1px solid #ddd;
#             border-radius: 4px;
#         }
#         .toggle-switch {
#             position: relative;
#             display: inline-block;
#             width: 60px;
#             height: 34px;
#         }
#         .toggle-switch input {
#             opacity: 0;
#             width: 0;
#             height: 0;
#         }
#         .slider {
#             position: absolute;
#             cursor: pointer;
#             top: 0;
#             left: 0;
#             right: 0;
#             bottom: 0;
#             background-color: #ccc;
#             transition: .4s;
#             border-radius: 34px;
#         }
#         .slider:before {
#             position: absolute;
#             content: "";
#             height: 26px;
#             width: 26px;
#             left: 4px;
#             bottom: 4px;
#             background-color: white;
#             transition: .4s;
#             border-radius: 50%;
#         }
#         input:checked + .slider {
#             background-color: #2196F3;
#         }
#         input:checked + .slider:before {
#             transform: translateX(26px);
#         }
#         .switch-label {
#             margin-left: 10px;
#             position: relative;
#             top: -10px;
#         }
#         .switch-container {
#             margin-bottom: 10px;
#         }
#         .highlight {
#             background-color: #f9f9c5;
#             padding: 10px;
#             border-left: 4px solid #ffd700;
#             margin-bottom: 10px;
#         }
#     </style>
# </head>
# <body>
#     <div class="container">
#         <h1>Ball Tracking System</h1>
        
#         <div class="controls">
#             <h2>Video Controls</h2>
#             <div class="control-group">
#                 <input type="text" id="video-path" placeholder="Enter path to video file" value="videos/7.mp4">
#                 <button id="open-video">Open Video</button>
#                 <button id="toggle-pause">Pause/Play</button>
#                 <button id="save-data">Save Tracking Data</button>
#                 <button id="stop-video">Stop Video</button>
#             </div>
#         </div>
        
#         <div class="controls">
#             <h2>Bowling Detection</h2>
#             <div class="highlight">
#                 <strong>New Feature:</strong> The system now detects new bowls and can clear previous ball paths automatically.
#             </div>
#             <div class="switch-container">
#                 <label class="toggle-switch">
#                     <input type="checkbox" id="toggle-clear-paths" checked>
#                     <span class="slider"></span>
#                 </label>
#                 <span class="switch-label">Clear paths on new bowls</span>
#             </div>
#         </div>
        
#         <div class="controls">
#             <h2>Overlay Controls</h2>
#             <div class="control-group">
#                 <div class="switch-container">
#                     <label class="toggle-switch">
#                         <input type="checkbox" id="toggle-bounding-boxes" checked>
#                         <span class="slider"></span>
#                     </label>
#                     <span class="switch-label">Bounding Boxes</span>
#                 </div>
                
#                 <div class="switch-container">
#                     <label class="toggle-switch">
#                         <input type="checkbox" id="toggle-centroids" checked>
#                         <span class="slider"></span>
#                     </label>
#                     <span class="switch-label">Centroids</span>
#                 </div>
                
#                 <div class="switch-container">
#                     <label class="toggle-switch">
#                         <input type="checkbox" id="toggle-trajectories" checked>
#                         <span class="slider"></span>
#                     </label>
#                     <span class="switch-label">Trajectories</span>
#                 </div>
                
#                 <div class="switch-container">
#                     <label class="toggle-switch">
#                         <input type="checkbox" id="toggle-angles" checked>
#                         <span class="slider"></span>
#                     </label>
#                     <span class="switch-label">Angles</span>
#                 </div>
                
#                 <div class="switch-container">
#                     <label class="toggle-switch">
#                         <input type="checkbox" id="toggle-ball-ids" checked>
#                         <span class="slider"></span>
#                     </label>
#                     <span class="switch-label">Ball IDs</span>
#                 </div>
                
#                 <div class="switch-container">
#                     <label class="toggle-switch">
#                         <input type="checkbox" id="toggle-stats" checked>
#                         <span class="slider"></span>
#                     </label>
#                     <span class="switch-label">Statistics</span>
#                 </div>
#             </div>
#         </div>
        
#         <div class="video-container">
#             <img id="video-feed" src="/video_feed" alt="Video Feed">
#         </div>
#     </div>
    
#     <script>
#         document.getElementById('open-video').addEventListener('click', function() {
#             const videoPath = document.getElementById('video-path').value;
#             fetch('/open_video', {
#                 method: 'POST',
#                 headers: {
#                     'Content-Type': 'application/json',
#                 },
#                 body: JSON.stringify({ video_path: videoPath }),
#             })
#             .then(response => response.json())
#             .then(data => {
#                 if (data.success) {
#                     console.log('Video opened:', data.video_info);
#                     // Reset video feed src to refresh stream
#                     document.getElementById('video-feed').src = '/video_feed?' + new Date().getTime();
#                 } else {
#                     alert('Error: ' + data.error);
#                 }
#             });
#         });
        
#         document.getElementById('toggle-pause').addEventListener('click', function() {
#             fetch('/toggle_pause', {
#                 method: 'POST',
#                 headers: {
#                     'Content-Type': 'application/json',
#                 },
#                 body: JSON.stringify({}),
#             })
#             .then(response => response.json())
#             .then(data => {
#                 console.log('Pause toggled:', data);
#             });
#         });
        
#         document.getElementById('save-data').addEventListener('click', function() {
#             const outputPath = prompt('Enter output path for tracking data:', 'tracking_data.json');
#             if (outputPath) {
#                 fetch('/save_data', {
#                     method: 'POST',
#                     headers: {
#                         'Content-Type': 'application/json',
#                     },
#                     body: JSON.stringify({ output_path: outputPath }),
#                 })
#                 .then(response => response.json())
#                 .then(data => {
#                     if (data.success) {
#                         alert('Tracking data saved successfully');
#                     } else {
#                         alert('Error saving tracking data');
#                     }
#                 });
#             }
#         });
        
#         document.getElementById('stop-video').addEventListener('click', function() {
#             fetch('/stop_video', {
#                 method: 'POST',
#                 headers: {
#                     'Content-Type': 'application/json',
#                 },
#                 body: JSON.stringify({}),
#             })
#             .then(response => response.json())
#             .then(data => {
#                 console.log('Video stopped:', data);
#             });
#         });
        
#         // Setup toggle switches
#         function setupToggleSwitch(id, endpoint, setting) {
#             const toggle = document.getElementById(id);
#             toggle.addEventListener('change', function() {
#                 fetch('/toggle_overlay', {
#                     method: 'POST',
#                     headers: {
#                         'Content-Type': 'application/json',
#                     },
#                     body: JSON.stringify({ setting: setting, value: this.checked }),
#                 })
#                 .then(response => response.json())
#                 .then(data => {
#                     console.log(`${setting} toggled:`, data);
#                 });
#             });
#         }

#         // Setup all toggle switches
#         setupToggleSwitch('toggle-bounding-boxes', '/toggle_overlay', 'bounding_boxes');
#         setupToggleSwitch('toggle-centroids', '/toggle_overlay', 'centroids');
#         setupToggleSwitch('toggle-trajectories', '/toggle_overlay', 'trajectories');
#         setupToggleSwitch('toggle-angles', '/toggle_overlay', 'angles');
#         setupToggleSwitch('toggle-ball-ids', '/toggle_overlay', 'ball_ids');
#         setupToggleSwitch('toggle-stats', '/toggle_overlay', 'stats');
#         setupToggleSwitch('toggle-clear-paths', '/toggle_overlay', 'clear_paths');

#         // Auto-start with default video when page loads
#         document.addEventListener('DOMContentLoaded', function() {
#             const defaultVideoPath = document.getElementById('video-path').value;
#             if (defaultVideoPath) {
#                 fetch('/open_video', {
#                     method: 'POST',
#                     headers: {
#                         'Content-Type': 'application/json',
#                     },
#                     body: JSON.stringify({ video_path: defaultVideoPath }),
#                 })
#                 .then(response => response.json())
#                 .then(data => {
#                     if (data.success) {
#                         console.log('Default video opened:', data.video_info);
#                     } else {
#                         console.error('Error opening default video:', data.error);
#                     }
#                 });
#             }
#         });
#     </script>
# </body>
# </html>
# ''')

# if __name__ == '__main__':
#     create_templates()
#     app.run(debug=True)

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
                
                # Add to current ball centroids
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
        tracking_data["frames"][frame_idx] = frame_data
        
        # Update for next iteration
        frame_idx += 1
        
        # Show progress
        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx}/{frame_count} frames ({(frame_idx/frame_count*100):.1f}%)")
    
    # Add the last ball trajectory if it exists
    if len(current_ball_centroids) > 0:
        tracking_data["balls"].append(current_ball_centroids)
    
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
    
    # Extract ball trajectories
    all_balls = tracking_data["balls"]
    
    # List to store active balls (balls currently visible in the frame)
    active_balls = []
    
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
        
        # Check for new balls that have started in this frame
        for ball_idx, ball in enumerate(all_balls):
            ball_start_frame = min([c[2] for c in ball]) if ball else 0
            ball_end_frame = max([c[2] for c in ball]) if ball else 0
            
            # If this ball is active in the current frame range, add it to active balls
            if ball_start_frame <= frame_idx <= ball_end_frame:
                if ball_idx not in [b[0] for b in active_balls]:
                    active_balls.append((ball_idx, ball))
            
            # If this ball is now past its end frame, remove it from active balls
            if frame_idx > ball_end_frame and ball_idx in [b[0] for b in active_balls]:
                active_balls = [b for b in active_balls if b[0] != ball_idx]
        
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
        cv2.imshow('Ball Tracking with Multi-Ball Support', frame)
        
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
    video_path = os.path.join('videos', '7.mp4')
    model_path = os.path.join('runs', 'detect', 'train9', 'weights', 'best.pt')
    tracking_data_path = 'ball_tracking_data.json'
    
    # First process the video and save tracking data
    # process_video(video_path, tracking_data_path,  model_path)
    
    # Then replay with overlay
    replay_with_overlay(video_path, tracking_data_path)