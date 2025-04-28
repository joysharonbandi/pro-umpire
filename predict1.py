from collections import deque
from ultralytics import YOLO
import math
import time
import cv2
import os

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    import math
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

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
    frame_resized = cv2.resize(frame, (480, 848))
    
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
            
            
                
    cv2.namedWindow('Calibration')
    cv2.setMouseCallback('Calibration', mouse_callback)
    cv2.imshow('Calibration', frame_resized)
    
    while len(points) <1:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Calculate pixels per meter based on user input
    # if len(points) == 1:
    #     pixel_distance = math.sqrt((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2)
        
        # Ask user for the real-world distance
       
    
    cap.release()
    cv2.destroyAllWindows()
    return points


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



# cap.release()
# cv2.destroyAllWindows()




def process_video(points):
    model_path = os.path.join('runs','detect','train9','weights','best.pt')
    model = YOLO(model_path)

    video_path = os.path.join('videos','ball.MOV')
    cap = cv2.VideoCapture(video_path)
    ret = True
    prevTime = 0
    centroid_history = FixedSizeQueue(10)
    start_time = time.time()
    interval = 0.6
    paused = False
    angle = 0
    prev_frame_time = 0 
    new_frame_time = 0
    distance_history = []
    max_distance_percent = 0
    max_distance_position = None
    bowler_position = None  # Will be set once detected
    batsman_position = None  # Will be set once detected
    pitch_length = 22 * 0.9144  # Standard cricket pitch length in meters (22 yards)
    pixel_to_meter_ratio = None  # Will be calculated based on pitch markers or assumptions
    estimated_3d_distance = None





    while ret:
        ret, frame = cap.read()
        if ret:
            new_frame_time = time.time() 
            fps = 1/(new_frame_time-prev_frame_time) 
            prev_frame_time = new_frame_time 
            fps = int(fps)  
            fps = str(fps)
            print(list(centroid_history.queue))
            current_time = time.time()
            if current_time - start_time >= interval and len(centroid_history)>0:
                centroid_history.pop()
                start_time = current_time
            
            results = model.track(frame, persist=True,conf=0.35,verbose=False)
            boxes = results[0].boxes
            box = boxes.xyxy
            rows,cols = box.shape
            if len(box)!=0:
                for i in range(rows):
                    x1,y1,x2,y2 = box[i]
                    x1,y1,x2,y2 = x1.item(),y1.item(),x2.item(),y2.item()
                    
                    centroid_x = int((x1+x2)/2)
                    centroid_y = int((y1+y2)/2)
                    
                    centroid_history.add((centroid_x, centroid_y))
                    cv2.circle(frame,(centroid_x, centroid_y),radius=3,color=(0,0,255),thickness=-1)
                    cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)

            if len(box) != 0:
                for i in range(rows):
                    frame_height = frame.shape[0]
                    top_portion_height = frame_height * 0.70

                    if centroid_y <= top_portion_height:
                        distance_from_top = centroid_y
                        distance_percent = (distance_from_top / top_portion_height) * 100
                        distance_history.append(distance_percent)

                        if distance_percent > max_distance_percent:
                            max_distance_percent = distance_percent
                            max_distance_position = (centroid_x, centroid_y)

                        # Print logs to console
                        print(f"Distance from top: {distance_percent:.2f}%")

                        # Display on frame
                        cv2.putText(frame, f"Distance: {distance_percent:.2f}%", (20, 80), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
                        cv2.putText(frame, f"Max Distance: {max_distance_percent:.2f}%", (20, 110), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

                        # Draw normal centroid in red
                        cv2.circle(frame, (centroid_x, centroid_y), radius=3, color=(0, 0, 255), thickness=-1)

                        # Mark the max distance position with brown color if it exists
                        if max_distance_position:
                            
                            cv2.circle(frame, max_distance_position, radius=5, color=(0, 75, 150), thickness=-1)  # Brown in BGR

                    # Draw a horizontal line at 75% of frame height to visualize the boundary
                    cv2.line(frame, (0, int(top_portion_height)), (frame.shape[1], int(top_portion_height)), (255, 255, 0), 1)

            # if len(centroid_history) > 1:
            #     centroid_list = list(centroid_history.get_queue())
            #     for i in range(1, len(centroid_history)):
            #         # if math.sqrt(y_diff**2+x_diff**2)<7:
            #         cv2.line(frame, centroid_history.get_queue()[i-1], centroid_history.get_queue()[i], (255, 0, 0), 4)    
                    
            # if len(centroid_history) > 1:
            #     centroid_list = list(centroid_history.get_queue())
            #     x_diff = centroid_list[-1][0] - centroid_list[-2][0]
            #     y_diff = centroid_list[-1][1] - centroid_list[-2][1]
            #     if(x_diff!=0):
            #         m1 = y_diff/x_diff
            #         if m1==1:
            #             angle = 90
            #         elif m1!=0:
            #             angle = 90-angle_between_lines(m1)
            #         if angle>=45:
            #                 print("ball bounced")
            #     future_positions = [centroid_list[-1]]
            #     for i in range(1, 5):
            #         future_positions.append(
            #             (
            #                 centroid_list[-1][0] + x_diff * i,
            #                 centroid_list[-1][1] + y_diff * i
            #             )
            #         )
            #     print("Future Positions: ",future_positions)
            #     for i in range(1,len(future_positions)):
            #         cv2.line(frame, future_positions[i-1], future_positions[i], (0, 255, 0), 4)
            #         cv2.circle(frame,future_positions[i],radius=3,color=(0,0,255),thickness=-1)

            text = "Angle: {:.2f} degrees".format(angle)
            cv2.putText(frame,text,(20,20),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
            cv2.putText(frame, f'FPS: {fps}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0), 2) 
            frame_resized = cv2.resize(frame, (1000, 600))
            cv2.imshow('frame',frame_resized)
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord(' '):
                paused = not paused
                
                while paused:
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord(' '):
                        paused = not paused
                    elif key == ord('q'):
                        break
    print(max_distance_position,"max distance frame",calculate_distance(points[0],max_distance_position))
    cap.release()
    cv2.destroyAllWindows()





if __name__ == "__main__":
    points=calibrate_pixels_per_meter('videos/7.mp4')
    print(points)
    process_video(points)
