from ultralytics import YOLO
import cv2
import os

model_path_1 = os.path.join('runs','detect','train5','weights','best.pt')
ball_model = YOLO(model_path_1)
model_path_2 = os.path.join('runs','detect','train8','weights','best.pt')
stump_model = YOLO(model_path_2)
# Load all models
# ball_model = YOLO("ball_detection.pt")
# stump_model = YOLO("stump_detection.pt")
# batsman_model = YOLO("batsman_pose.pt")



def detect_all(frame):
    ball_results = ball_model.track(frame, persist=True)
    stump_results = stump_model(frame)
    # batsman_results = batsman_model(frame)
    
    return {
        "ball": ball_results[0].boxes.xyxy,  # [x1,y1,x2,y2]
        "stumps": stump_results[0].boxes.xywh,  # [center_x, center_y, width, height]
        # "batsman": batsman_results[0].keypoints  # Pose landmarks
    }


def get_homography(stumps_detected):
    # Real-world stump positions (cm)
    stumps_real = np.float32([[0, 0], [22.86, 0], [45.72, 0]])
    
    # Detected stumps (sorted left-to-right)
    stumps_detected = sorted(stumps_detected, key=lambda x: x[0]) 
    stumps_pixels = np.float32(stumps_detected)
    
    # Calculate homography
    matrix, _ = cv2.findHomography(stumps_pixels, stumps_real)
    return matrix

# Usage
# detections = detect_all(frame)
# matrix = get_homography(detections["stumps"])



from filterpy.kalman import KalmanFilter
import numpy as np

class BallTracker:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=6, dim_z=2)
        # State: [x, y, vx, vy, ax, ay]
        self.kf.F = np.array([[1,0,1,0,0.5,0],  # State transition
                             [0,1,0,1,0,0.5],
                             [0,0,1,0,1,0],
                             [0,0,0,1,0,1],
                             [0,0,0,0,1,0],
                             [0,0,0,0,0,1]])
        
    def update(self, x, y):
        self.kf.predict()
        self.kf.update(np.array([[x], [y]]))
        
    def predict_next(self):
        return self.kf.x[:2].flatten()

# Initialize
tracker = BallTracker()


def check_lbw(ball_pos, batsman_foot, stumps_real):
    # 1. Check impact in line with stumps
    impact_in_line = (0 <= ball_pos[0] <= 22.86)
    
    # 2. Check batsman's foot position (simplified)
    foot_in_crease = (batsman_foot[0] > 0)  # 0 = crease line
    
    # 3. Predict if ball hits stumps
    ball_moving_towards_stumps = (ball_pos[1] < 10)  # 10cm from stumps
    
    return impact_in_line and not foot_in_crease and ball_moving_towards_stumps






video_path = os.path.join('videos','5.mp4')
cap = cv2.VideoCapture(video_path)
tracker = BallTracker()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Step 1: Detect all objects
    detections = detect_all(frame)
    
    # Step 2: Track ball with Kalman filter
    ball_x, ball_y = get_centroid(detections["ball"][0])
    tracker.update(ball_x, ball_y)
    predicted_pos = tracker.predict_next()
    
    # Step 3: Project to real-world coordinates
    ball_real = cv2.perspectiveTransform(np.array([[[ball_x, ball_y]]], dtype=np.float32), matrix)
    
    # Step 4: LBW Check
    batsman_foot = detections["batsman"].xy[0][15]  # Landmark 15 = right foot
    lbw_out = check_lbw(ball_real, batsman_foot, stumps_real)
    
    # Visualization
    cv2.putText(frame, f"LBW: {'OUT' if lbw_out else 'NOT OUT'}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("DRS System", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()