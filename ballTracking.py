# # from collections import deque
# # from ultralytics import YOLO
# # import math
# # import time
# # import cv2
# # import os

# # def angle_between_lines(m1, m2=1):
# #     if m1 != -1/m2:
# #         angle = math.degrees(math.atan(abs((m2 - m1) / (1 + m1 * m2))))
# #         return angle
# #     else:
# #         return 90.0

# # class FixedSizeQueue:
# #     def __init__(self, max_size):
# #         self.queue = deque(maxlen=max_size)
    
# #     def add(self, item):
# #         self.queue.append(item)
    
# #     def pop(self):
# #         self.queue.popleft()
        
# #     def clear(self):
# #         self.queue.clear()

# #     def get_queue(self):
# #         return self.queue
    
# #     def __len__(self):
# #         return len(self.queue)


# # model_path = os.path.join('runs','detect','train5','weights','best.pt')
# # model = YOLO(model_path)

# # video_path = os.path.join('videos','7.mp4')
# # cap = cv2.VideoCapture(video_path)
# # ret = True
# # prevTime = 0
# # centroid_history = FixedSizeQueue(10)
# # start_time = time.time()
# # interval = 0.6
# # paused = False
# # angle = 0
# # prev_frame_time = 0 
# # new_frame_time = 0

# # while ret:
# #     ret, frame = cap.read()
# #     if ret:
# #         new_frame_time = time.time() 
# #         fps = 1/(new_frame_time-prev_frame_time) 
# #         prev_frame_time = new_frame_time 
# #         fps = int(fps)  
# #         fps = str(fps)
# #         print(list(centroid_history.queue))
# #         current_time = time.time()
# #         if current_time - start_time >= interval and len(centroid_history)>0:
# #             centroid_history.pop()
# #             start_time = current_time
        
# #         results = model.track(frame, persist=True,conf=0.35,verbose=False)
# #         boxes = results[0].boxes
# #         box = boxes.xyxy
# #         rows,cols = box.shape
# #         if len(box)!=0:
# #             for i in range(rows):
# #                 x1,y1,x2,y2 = box[i]
# #                 x1,y1,x2,y2 = x1.item(),y1.item(),x2.item(),y2.item()
                
# #                 centroid_x = int((x1+x2)/2)
# #                 centroid_y = int((y1+y2)/2)
                
# #                 centroid_history.add((centroid_x, centroid_y))
# #                 cv2.circle(frame,(centroid_x, centroid_y),radius=3,color=(0,0,255),thickness=-1)
# #                 cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)
                
# #         # if len(centroid_history) > 1:
# #         #     centroid_list = list(centroid_history.get_queue())
# #         #     for i in range(1, len(centroid_history)):
# #         #         # if math.sqrt(y_diff**2+x_diff**2)<7:
# #         #         cv2.line(frame, centroid_history.get_queue()[i-1], centroid_history.get_queue()[i], (255, 0, 0), 4)    
                
# #         if len(centroid_history) > 1:
# #             centroid_list = list(centroid_history.get_queue())
# #             x_diff = centroid_list[-1][0] - centroid_list[-2][0]
# #             y_diff = centroid_list[-1][1] - centroid_list[-2][1]
# #             if(x_diff!=0):
# #                 m1 = y_diff/x_diff
# #                 if m1==1:
# #                     angle = 90
# #                 elif m1!=0:
# #                     angle = 90-angle_between_lines(m1)
# #                 if angle>=45:
# #                         print("ball bounced")
# #             future_positions = [centroid_list[-1]]
# #             for i in range(1, 5):
# #                 future_positions.append(
# #                     (
# #                         centroid_list[-1][0] + x_diff * i,
# #                         centroid_list[-1][1] + y_diff * i
# #                     )
# #                 )
# #             print("Future Positions: ",future_positions)
# #             for i in range(1,len(future_positions)):
# #                 cv2.line(frame, future_positions[i-1], future_positions[i], (0, 255, 0), 4)
# #                 cv2.circle(frame,future_positions[i],radius=3,color=(0,0,255),thickness=-1)
                

# #         text = "Angle: {:.2f} degrees".format(angle)
# #         cv2.putText(frame,text,(20,20),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
# #         cv2.putText(frame, f'FPS: {fps}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0), 2) 
# #         frame_resized = cv2.resize(frame, (1000, 600))
# #         cv2.imshow('frame',frame_resized)
         
# #         key = cv2.waitKey(1)
# #         if key & 0xFF == ord('q'):
# #             break
# #         elif key & 0xFF == ord(' '):
# #             paused = not paused
            
# #             while paused:
# #                 key = cv2.waitKey(30) & 0xFF
# #                 if key == ord(' '):
# #                     paused = not paused
# #                 elif key == ord('q'):
# #                     break
# # cap.release()
# # cv2.destroyAllWindows()




# # # import numpy as np
# # # import cv2
# # # import time
# # # import os
# # # import math
# # # from collections import deque
# # # from ultralytics import YOLO


# # # def angle_between_lines(m1, m2=1):
# # #     if m1 != -1/m2:
# # #         angle = math.degrees(math.atan(abs((m2 - m1) / (1 + m1 * m2))))
# # #         return angle
# # #     else:
# # #         return 90.0


# # # class FixedSizeQueue:
# # #     def __init__(self, max_size):
# # #         self.queue = deque(maxlen=max_size)
    
# # #     def add(self, item):
# # #         self.queue.append(item)
    
# # #     def pop(self):
# # #         self.queue.popleft()

# # #     def clear(self):
# # #         self.queue.clear()

# # #     def get_queue(self):
# # #         return self.queue
    
# # #     def __len__(self):
# # #         return len(self.queue)


# # # model_path = os.path.join('runs', 'detect', 'train5', 'weights', 'best.pt')
# # # model = YOLO(model_path)

# # # video_path = os.path.join('videos', 'test1.mp4')
# # # cap = cv2.VideoCapture(video_path)
# # # ret = True
# # # prevTime = 0
# # # centroid_history = FixedSizeQueue(10)
# # # start_time = time.time()
# # # interval = 0.6
# # # paused = False
# # # angle = 0
# # # prev_frame_time = 0 
# # # new_frame_time = 0

# # # # Smoothing function for lines (Bezier curve)
# # # def create_bezier_curve(points, smoothness=50):
# # #     t = np.linspace(0, 1, smoothness)
# # #     curve = []
# # #     for i in range(smoothness):
# # #         x = (1 - t[i]) ** 2 * points[0][0] + 2 * (1 - t[i]) * t[i] * points[1][0] + t[i] ** 2 * points[2][0]
# # #         y = (1 - t[i]) ** 2 * points[0][1] + 2 * (1 - t[i]) * t[i] * points[1][1] + t[i] ** 2 * points[2][1]
# # #         curve.append([int(x), int(y)])
# # #     return np.array(curve, dtype=np.int32)


# # # while ret:
# # #     ret, frame = cap.read()
# # #     if ret:
# # #         new_frame_time = time.time() 
# # #         fps = 1/(new_frame_time-prev_frame_time) 
# # #         prev_frame_time = new_frame_time 
# # #         fps = int(fps)  
# # #         fps = str(fps)
        
# # #         current_time = time.time()
# # #         if current_time - start_time >= interval and len(centroid_history) > 0:
# # #             centroid_history.pop()
# # #             start_time = current_time
        
# # #         results = model.track(frame, persist=True, conf=0.35, verbose=False)
# # #         boxes = results[0].boxes
# # #         box = boxes.xyxy
# # #         rows, cols = box.shape
# # #         if len(box) != 0:
# # #             for i in range(rows):
# # #                 x1, y1, x2, y2 = box[i]
# # #                 x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
                
# # #                 centroid_x = int((x1 + x2) / 2)
# # #                 centroid_y = int((y1 + y2) / 2)
                
# # #                 centroid_history.add((centroid_x, centroid_y))
# # #                 cv2.circle(frame, (centroid_x, centroid_y), radius=3, color=(0, 0, 255), thickness=-1)
# # #                 cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                
# # #         # Smoothly connecting centroids using Bezier curve
# # #         if len(centroid_history) > 2:
# # #             centroid_list = list(centroid_history.get_queue())
# # #             curve_points = []
# # #             for i in range(1, len(centroid_history)):
# # #                 mid_point = (
# # #                     int((centroid_list[i - 1][0] + centroid_list[i][0]) / 2),
# # #                     int((centroid_list[i - 1][1] + centroid_list[i][1]) / 2)
# # #                 )
# # #                 curve_points.append(mid_point)
            
# # #             bezier_curve = create_bezier_curve([centroid_list[0], curve_points[0], centroid_list[-1]])
# # #             cv2.polylines(frame, [bezier_curve], isClosed=False, color=(255, 0, 0), thickness=3)
            
# # #         # Calculate angle and future positions
# # #         if len(centroid_history) > 1:
# # #             centroid_list = list(centroid_history.get_queue())
# # #             x_diff = centroid_list[-1][0] - centroid_list[-2][0]
# # #             y_diff = centroid_list[-1][1] - centroid_list[-2][1]
            
# # #             if x_diff != 0:
# # #                 m1 = y_diff / x_diff
# # #                 if m1 == 1:
# # #                     angle = 90
# # #                 elif m1 != 0:
# # #                     angle = 90 - angle_between_lines(m1)
            
# # #             future_positions = [centroid_list[-1]]
# # #             for i in range(1, 5):
# # #                 future_positions.append(
# # #                     (
# # #                         centroid_list[-1][0] + x_diff * i,
# # #                         centroid_list[-1][1] + y_diff * i
# # #                     )
# # #                 )
            
# # #             # Smoothly connect future positions
# # #             bezier_curve_future = create_bezier_curve([future_positions[0], future_positions[1], future_positions[-1]])
# # #             cv2.polylines(frame, [bezier_curve_future], isClosed=False, color=(0, 255, 0), thickness=3)
            
# # #             for pos in future_positions:
# # #                 cv2.circle(frame, pos, radius=3, color=(0, 0, 255), thickness=-1)
                
# # #         text = "Angle: {:.2f} degrees".format(angle)
# # #         cv2.putText(frame, text, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
# # #         cv2.putText(frame, f'FPS: {fps}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) 
        
# # #         frame_resized = cv2.resize(frame, (1000, 600))
# # #         cv2.imshow('frame', frame_resized)
         
# # #         key = cv2.waitKey(1)
# # #         if key & 0xFF == ord('q'):
# # #             break
# # #         elif key & 0xFF == ord(' '):
# # #             paused = not paused
            
# # #             while paused:
# # #                 key = cv2.waitKey(30) & 0xFF
# # #                 if key == ord(' '):
# # #                     paused = not paused
# # #                 elif key == ord('q'):
# # #                     break

# # # cap.release()
# # # cv2.destroyAllWindows()


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
#         "frames": {}
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
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Get tracking data for this frame
#         frame_data = tracking_data["frames"].get(str(frame_idx), {})
        
#         # Draw centroids and bounding boxes
#         for centroid in frame_data.get("centroids", []):
#             cv2.circle(frame, (centroid[0], centroid[1]), radius=3, color=(0, 0, 255), thickness=-1)
        
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
        
#         # Display FPS
#         cv2.putText(frame, f'FPS: {video_fps}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
#         # Resize and show frame
#         # frame_resized = cv2.resize(frame, (1000, 600))
#         cv2.imshow('frame', frame)
        
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
        
#         frame_idx += 1
    
#     cap.release()
#     cv2.destroyAllWindows()


# def predictObjects(video_path, model_path):
#     model = YOLO(model_path)
#     # cap = cv2.VideoCapture(video_path)
#     # ret = True
#     results = model(video_path, stream=True)  
    
#     print(results)

#     for r in results:
#         r.show()
#         boxes = r.boxes
#         for box in boxes:
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#             print(x1, y1, x2, y2)


# if __name__ == "__main__":
#     video_path = os.path.join('videos', 'ball.mov')
#     model_path = os.path.join('runs', 'detect', 'train11', 'best.pt')
#     tracking_data_path = 'ball_tracking_data.json'
#     predictObjects(video_path, model_path)
    
#     # Uncomment the one you want to run:
#     # process_video(video_path, tracking_data_path, model_path)  # First process and save data
#     # replay_with_overlay(video_path, tracking_data_path)  # Then replay with overlay






import numpy as np
from ultralytics import YOLO
import cv2
import math
import json


# def predictObjects(video_path, model_path):
#     import numpy as np
#     from ultralytics import YOLO
#     import cv2
#     import math
#     import json

#     model = YOLO(model_path)
#     cap = cv2.VideoCapture(video_path)
    
#     # Data structure to store tracking information
#     tracking_data = {
#         "frames": []
#     }
    
#     frame_count = 0
    
#     # Process the video frame by frame
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         # Create a copy of the frame for drawing
#         display_frame = frame.copy()
        
#         # Run YOLOv8 inference on the frame
#         results = model(frame)
        
#         # Initialize objects for this frame
#         frame_data = {
#             "frame_number": frame_count,
#             "objects": {},
#             "distances": {}
#         }
        
#         # Process detection results
#         for r in results:
#             boxes = r.boxes
#             for box in boxes:
#                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
#                 # Get class information
#                 class_id = int(box.cls[0])
#                 class_name = model.names[class_id]
#                 confidence = float(box.conf[0])
                
#                 # Calculate center point of the bounding box
#                 center_x = (x1 + x2) / 2
#                 center_y = (y1 + y2) / 2
                
#                 # Store object information
#                 if class_name not in frame_data["objects"]:
#                     frame_data["objects"][class_name] = []
                
#                 frame_data["objects"][class_name].append({
#                     "bbox": [x1, y1, x2, y2],
#                     "center": [center_x, center_y],
#                     "confidence": confidence
#                 })
                
#                 # Draw bounding box on the display frame
#                 cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(display_frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
#         # Calculate distances between objects
#         calculate_distances(frame_data)
        
#         # Store frame data
#         tracking_data["frames"].append(frame_data)
        
#         # Draw trajectory if ball is detected across frames
#         if frame_count > 0:
#             draw_ball_trajectory(tracking_data, display_frame, frame_count)
        
#         # Display the frame with drawings
#         cv2.imshow("Cricket Analysis", display_frame)
        
#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(30) & 0xFF == ord('q'):  # Increased wait time for better visibility
#             break
            
#         frame_count += 1
    
#     # Release resources
#     cap.release()
#     cv2.destroyAllWindows()
    
#     # Save tracking data to a file
#     with open('ball_tracking_data.json', 'w') as f:
#         json.dump(tracking_data, f)
    
#     return tracking_data

# def draw_ball_trajectory(tracking_data, display_frame, current_frame_idx):
    
#     """Draw the trajectory of the ball based on previous frames"""
#     # Get past frames to determine trajectory
#     max_trajectory_length = 20  # Increased for more visible trajectory
#     start_frame = max(0, current_frame_idx - max_trajectory_length)
    
#     ball_positions = []
#     for i in range(start_frame, current_frame_idx):
#         if i < len(tracking_data["frames"]):
#             frame_data = tracking_data["frames"][i]
#             if "ball" in frame_data["objects"] and frame_data["objects"]["ball"]:
#                 # Use the first detected ball (assuming there's only one)
#                 ball_pos = tuple(map(int, frame_data["objects"]["ball"][0]["center"]))
#                 ball_positions.append(ball_pos)
    
#     # Draw trajectory line if we have ball positions
#     if len(ball_positions) > 1:
#         # Check for start and end conditions based on distances
#         current_frame_data = tracking_data["frames"][current_frame_idx-1]
#         print(current_frame_data,"framedatsd")
        
#         # Start condition: Ball is near bowler
        
#         show_trajectory = False
#         if "distances" in current_frame_data and "ball_to_bowler" in current_frame_data["distances"]:
#             if current_frame_data["distances"]["ball_to_bowler"] < 200:  # Increased threshold for better detection
#                 show_trajectory = True
        
#         # End condition: Ball is near batsman or bat
#         end_trajectory = False
#         if "distances" in current_frame_data:
            
#             if "ball_to_batsman" in current_frame_data["distances"] and current_frame_data["distances"]["ball_to_batsman"] < 167:
#                 print(current_frame_data.get("distances",{}).get("ball_to_batsman"),"ball_to_batsman distance123")
#                 end_trajectory = True
#             elif "ball_to_bat" in current_frame_data["distances"] and current_frame_data["distances"]["ball_to_bat"] < 150:
#                 print(current_frame_data.get("distances","").get("ball_to_bat"), "ball_to_bat distance123")
#                 end_trajectory = True
        
#         # Always show trajectory for debugging (remove this condition later if needed)
#         show_trajectory = True

#         if(current_frame_data.get('frame_number',"")==111):
#             print("insdie the frmae data")
#             end_trajectory = True
#             show_trajectory= False
        
#         # Draw trajectory if conditions are met
#         if show_trajectory and not end_trajectory:
#     # Draw the trajectory lines
#             for i in range(1, len(ball_positions)):
#                 if(current_frame_data.get('frame_number',"")<=108):
                
#                     cv2.line(display_frame, ball_positions[i-1], ball_positions[i], (0, 255, 255), 3)
            
#             # Draw a circle at the current position of the ball
#             # if ball_positions:
#             #     cv2.circle(display_frame, ball_positions[-1], 7, (255, 0, 0), -1)
                
#             # # Optional: Draw start and end points more distinctly
#             # if len(ball_positions) > 2:
#             #     cv2.circle(display_frame, ball_positions[0], 10, (255, 255, 0), -1)  # Start point




def draw_ball_trajectory(tracking_data, display_frame, current_frame_idx):
    """Draw the trajectory of the ball with enhanced visualization based on distance analysis"""
    import cv2
    import numpy as np
    
    # Get past frames to determine trajectory
    max_trajectory_length = 30  # Maximum number of past frames to consider
    start_frame = max(0, current_frame_idx - max_trajectory_length)
    
    # Collect ball positions from past frames
    ball_positions = []
    frame_numbers = []
    
    for i in range(start_frame, current_frame_idx):
        if i < len(tracking_data["frames"]):
            frame_data = tracking_data["frames"][i]
            if "ball" in frame_data["objects"] and frame_data["objects"]["ball"]:
                # Use the first detected ball (assuming there's only one)
                ball_pos = tuple(map(int, frame_data["objects"]["ball"][0]["center"]))
                ball_positions.append(ball_pos)
                frame_numbers.append(frame_data["frame_number"])
    
    # If we have ball positions, process the trajectory
    if len(ball_positions) > 1:
        # Get current frame data
        current_frame_data = tracking_data["frames"][current_frame_idx-1]
        current_frame_number = current_frame_data["frame_number"]
        
        # Create a semi-transparent overlay for trajectories
        overlay = display_frame.copy()
        
        # Identify trajectory phases based on distances
        delivery_phase = []  # Ball from bowler to batsman
        return_phase = []    # Ball after being hit/played
        
        # Determine phase for each position
        for i, pos in enumerate(ball_positions):
            frame_idx = frame_numbers[i]
            
            # Find the corresponding frame data
            frame_data = None
            for f in tracking_data["frames"]:
                if f["frame_number"] == frame_idx:
                    frame_data = f
                    break
            
            if frame_data is None:
                continue
                
            # Get distances if available
            ball_to_bowler = frame_data.get("distances", {}).get("ball_to_bowler", 999)
            ball_to_batsman = frame_data.get("distances", {}).get("ball_to_batsman", 999)
            ball_to_bat = frame_data.get("distances", {}).get("ball_to_bat", 999)
            
            # Determine if this is delivery or return phase
            if frame_idx < 104:  # Before the key frame where ball likely made contact
                delivery_phase.append(pos)
            else:
                return_phase.append(pos)
        
        # Draw the delivery phase trajectory (bowler to batsman)
        if len(delivery_phase) > 1:
            for i in range(1, len(delivery_phase)):
                # Gradient from green (bowler) to yellow (batsman)
                progress = i / len(delivery_phase)
                color = (0, 255 * (1-progress), 255)  # BGR format: (Blue, Green, Red)
                thickness = 3
                
                cv2.line(overlay, delivery_phase[i-1], delivery_phase[i], color, thickness)
        
        # Draw the return phase trajectory (after hit/play)
        if len(return_phase) > 1:
            for i in range(1, len(return_phase)):
                # Use red for the return phase
                cv2.line(overlay, return_phase[i-1], return_phase[i], (0, 0, 255), 3)
        
        # Apply the semi-transparent overlay
        alpha = 0.7  # Transparency factor
        cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
        
        # PREDICT FUTURE TRAJECTORY AFTER FRAME 108
        if current_frame_number >= 108 and len(return_phase) >= 3:
            # Create a special overlay for predicted path
            future_overlay = display_frame.copy()
            
            # Make batsman semi-transparent in the overlay
            if "batsman" in current_frame_data["objects"] and current_frame_data["objects"]["batsman"]:
                for batsman in current_frame_data["objects"]["batsman"]:
                    x1, y1, x2, y2 = map(int, batsman["bbox"])
                    batsman_roi = future_overlay[y1:y2, x1:x2]
                    # Apply semi-transparency to the batsman region
                    batsman_transparent = cv2.addWeighted(batsman_roi, 0.3, np.zeros_like(batsman_roi), 0.7, 0)
                    future_overlay[y1:y2, x1:x2] = batsman_transparent
            
            # Use the latest positions to predict future trajectory
            if len(return_phase) >= 3:
                # Use the last 3 points to extrapolate future trajectory
                last_points = return_phase[-3:]
                
                # Calculate direction vector based on the last points
                dx = (last_points[-1][0] - last_points[-3][0]) / 2
                dy = (last_points[-1][1] - last_points[-3][1]) / 2
                
                # Predict future positions
                future_positions = []
                current_pos = last_points[-1]
                
                # Define the position of stumps (approximate - should be detected or defined)
                # Assuming stumps are positioned approximately at:
                stump_x = tracking_data["video_info"]["width"] * 0.75  # Adjust as needed
                stump_y = tracking_data["video_info"]["height"] * 0.6   # Adjust as needed
                stumps_width = 30  # Width of stumps in pixels
                stumps_height = 80  # Height of stumps in pixels
                
                # Draw stumps
                cv2.rectangle(future_overlay, 
                              (int(stump_x - stumps_width/2), int(stump_y - stumps_height)), 
                              (int(stump_x + stumps_width/2), int(stump_y)), 
                              (255, 255, 255), 2)
                cv2.putText(future_overlay, "Stumps", 
                          (int(stump_x - 30), int(stump_y - stumps_height - 10)),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Predict 15 future positions
                for i in range(15):
                    next_x = int(current_pos[0] + dx)
                    next_y = int(current_pos[1] + dy)
                    future_positions.append((next_x, next_y))
                    current_pos = (next_x, next_y)
                    
                    # Apply some gravity effect for more realistic trajectory
                    dy += 0.5  # Ball gradually falls faster
                
                # Draw the predicted trajectory
                if future_positions:
                    start_point = last_points[-1]
                    for i, point in enumerate(future_positions):
                        # Gradient from red to pink for future trajectory
                        alpha_factor = 1 - (i / len(future_positions)) * 0.7  # Decreasing opacity
                        
                        # Draw line with decreasing opacity
                        cv2.line(future_overlay, start_point, point, (0, 0, 255), 2)
                        
                        # Draw point with decreasing size
                        point_size = max(2, 6 - i // 3)
                        cv2.circle(future_overlay, point, point_size, (0, 0, 255), -1)
                        
                        start_point = point
                
                # Check if trajectory will hit stumps
                will_hit_stumps = False
                for point in future_positions:
                    if (stump_x - stumps_width/2 <= point[0] <= stump_x + stumps_width/2 and
                        stump_y - stumps_height <= point[1] <= stump_y):
                        will_hit_stumps = True
                        break
                
                # Display prediction result
                result_text = "PREDICTION: WILL HIT STUMPS!" if will_hit_stumps else "PREDICTION: WILL MISS STUMPS"
                result_color = (0, 0, 255) if will_hit_stumps else (0, 255, 0)  # Red if hit, green if miss
                
                cv2.putText(future_overlay, result_text, 
                          (50, 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, result_color, 2)
                
                # Apply the future trajectory overlay
                future_alpha = 0.8  # Higher transparency for the future prediction
                cv2.addWeighted(future_overlay, future_alpha, display_frame, 1 - future_alpha, 0, display_frame)
        
        # Draw ball position markers
        if delivery_phase:
            # Mark the start of the delivery (near bowler)
            cv2.circle(display_frame, delivery_phase[0], 8, (255, 255, 0), -1)  # Yellow
            cv2.putText(display_frame, "Release", 
                      (delivery_phase[0][0] - 30, delivery_phase[0][1] - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Mark the contact point if available (frame 102-103)
        contact_point = None
        min_bat_distance = float('inf')
        
        for f in tracking_data["frames"]:
            if "distances" in f and "ball_to_bat" in f["distances"]:
                if f["distances"]["ball_to_bat"] < min_bat_distance:
                    min_bat_distance = f["distances"]["ball_to_bat"]
                    for obj in f.get("objects", {}).get("ball", []):
                        if "center" in obj:
                            contact_point = tuple(map(int, obj["center"]))
        
        if contact_point:
            cv2.circle(display_frame, contact_point, 10, (0, 0, 255), -1)  # Red
            cv2.putText(display_frame, "Contact", 
                      (contact_point[0] + 10, contact_point[1] - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw the current ball position
        if ball_positions:
            cv2.circle(display_frame, ball_positions[-1], 6, (255, 0, 0), -1)
        
        # Display phase information
        display_phase_info(display_frame, current_frame_number)
        
        # Display distance measurements
        display_distance_measurements(display_frame, current_frame_data)

# Update the display_phase_info function to include a new prediction phase
def display_phase_info(display_frame, current_frame_number):
    """Display the current phase of the ball trajectory"""
    phase_text = ""
    
    if 90 <= current_frame_number < 98:
        phase_text = "Pre-Delivery"
    elif 98 <= current_frame_number < 104:
        phase_text = "Ball Delivery"
    elif 104 <= current_frame_number < 108:
        phase_text = "Post-Contact"
    elif current_frame_number >= 108:
        phase_text = "Trajectory Prediction"
    elif 110 <= current_frame_number < 117:
        phase_text = "Ball Settling"
    else:
        phase_text = "Ball Return"
    
    # Add phase information at the bottom of the frame
    h, w = display_frame.shape[:2]
    cv2.rectangle(display_frame, (10, h-40), (250, h-10), (0, 0, 0), -1)
    cv2.putText(display_frame, f"Phase: {phase_text}", (20, h-20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def display_phase_info(display_frame, current_frame_number):
    """Display the current phase of the ball trajectory"""
    phase_text = ""
    
    if 90 <= current_frame_number < 98:
        phase_text = "Pre-Delivery"
    elif 98 <= current_frame_number < 104:
        phase_text = "Ball Delivery"
    elif 104 <= current_frame_number < 110:
        phase_text = "Post-Contact"
    elif 110 <= current_frame_number < 117:
        phase_text = "Ball Settling"
    else:
        phase_text = "Ball Return"
    
    # Add phase information at the bottom of the frame
    h, w = display_frame.shape[:2]
    cv2.rectangle(display_frame, (10, h-40), (250, h-10), (0, 0, 0), -1)
    cv2.putText(display_frame, f"Phase: {phase_text}", (20, h-20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def display_distance_measurements(display_frame, current_frame_data):
    """Display the distance measurements on the frame"""
    import cv2
    
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
        if dist < 170:
            cv2.circle(display_frame, (330, 95), 8, (0, 255, 0), -1)  # Green when close
        else:
            cv2.circle(display_frame, (330, 95), 8, (0, 0, 255), -1)  # Red when far
    
    # Display Ball to Bat distance
    if "distances" in current_frame_data and "ball_to_bat" in current_frame_data["distances"]:
        dist = current_frame_data["distances"]["ball_to_bat"]
        cv2.putText(display_frame, f"Ball to Bat: {dist:.1f}", (20, 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Color-coded indicator
        if dist < 60:
            cv2.circle(display_frame, (330, 125), 8, (0, 255, 0), -1)  # Green when close
        else:
            cv2.circle(display_frame, (330, 125), 8, (0, 0, 255), -1)  # Red when far



import time

def predictObjects(video_path, model_path):
    # import numpy as np
    # from ultralytics import YOLO
    # import cv2
    # import math
    # import json
    # import time
    paused = False
    

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    
    # Data structure to store distance information
    distance_data = {
        "frames": []
    }
    
    # Full tracking data for visualization
    tracking_data = {
           "video_info": {
            "fps": fps,
            "frame_count": frame_count,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        },
          "frames": [],
    }
    
    frame_count = 0
    video_fps = tracking_data["video_info"]["fps"]
    frame_delay = int(1000 / video_fps) 
    
    # Process the video frame by frame
    while cap.isOpened():
        
        ret, frame = cap.read()
        if not ret:
            break
            
        # Create a copy of the frame for drawing
        display_frame = frame.copy()
        
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        # Initialize objects for this frame
        frame_data = {
            "frame_number": frame_count,
            "objects": {},
            "distances": {}
        }

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
        
        # Process detection results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get class information
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                print(class_name,"cjh")
                confidence = float(box.conf[0])
                
                # Calculate center point of the bounding box
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
                
                # Draw bounding box on the display frame
                # if(class_name == "bowlers_hand"):
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Calculate distances between objects
        calculate_distances(frame_data)
        
        # Extract only the distance data for the clean output
        frame_distance_data = {
            "frame_number": frame_count,
            "timestamp": frame_count / cap.get(cv2.CAP_PROP_FPS),  # Convert to seconds
            "distances": {}
        }
        
        # Only include non-None distances
        if "distances" in frame_data:
            for key, value in frame_data["distances"].items():
                if value is not None:
                    frame_distance_data["distances"][key] = value
        
        # Only add frames that have distance data
        if frame_distance_data["distances"]:
            distance_data["frames"].append(frame_distance_data)
        
        # Store frame data
        tracking_data["frames"].append(frame_data)
        
        # Draw trajectory if ball is detected across frames
        # if frame_count > 0:
            # draw_ball_trajectory(tracking_data, display_frame, frame_count)
        
        # Display the frame with drawings
        cv2.imshow("Cricket Analysis", display_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
            
        frame_count += 1
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Add metadata to distance data
    distance_data["metadata"] = {
        "video_path": video_path,
        "model_path": model_path,
        "total_frames": frame_count,
        # "video_duration": frame_count / cap.get(cv2.CAP_PROP_FPS) if frame_count > 0 else 0,
        "date_processed": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save distance data to a clean JSON file
    with open('ball_distance_data.json', 'w') as f:
        json.dump(distance_data, f, indent=2)
    
    # Save full tracking data to a file
    with open('ball_tracking_data.json', 'w') as f:
        json.dump(tracking_data, f)
    
    print(f"Distance data saved to 'ball_distance_data.json'")
    print(f"Full tracking data saved to 'ball_tracking_data.json'")
    
    return distance_data

def calculate_distances(frame_data):
    """Calculate distances between different objects in the frame"""
    # Check if we have the objects we need
    objects = frame_data["objects"]


    if "ball" in objects and "bowler_hand" in objects:
        print(distance,"bowler_hand distance")
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

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    import math
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)









if __name__ == "__main__":
    import os
    from ultralytics import YOLO
    
    video_path = os.path.join('videos', 'ball.mov')
    model_path = os.path.join('runs', 'detect', 'train11', 'best.pt')
    tracking_data = predictObjects(video_path, model_path)