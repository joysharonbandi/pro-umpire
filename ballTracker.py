import cv2
import numpy as np
import math
import os
from sklearn.svm import SVC
from sklearn import linear_model
import vpython as vp

class CricketUmpireSystem:
    def __init__(self):
        self.ball_detector = BallDetector()
        self.batsman_detector = BatsmanDetector()
        self.ball_tracker = BallTracker()
        self.coordinator_3d = Coordinator3D()
        self.decision_maker = DecisionMaker()
        self.visualizer = Visualizer3D()
        
    def process_video(self, video_path, action_type='fast'):
        """Process cricket video to provide umpire assistance"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
            
        frame_count = 0
        ball_positions = []
        batsman_positions = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every 3rd frame for performance
            if frame_count % 3 == 0:
                # Step 1: Ball Detection
                ball_window = self.ball_detector.detect(frame)
                
                # Step 2: Ball Tracking
                if ball_window is not None:
                    ball_info = self.ball_tracker.track(frame, ball_window)
                    if ball_info:
                        ball_positions.append(ball_info)
                
                # Step 3: Batsman Detection
                batsman_window = self.batsman_detector.detect(frame)
                if batsman_window is not None:
                    batsman_info = {'x': batsman_window[0], 'y': batsman_window[1], 
                                    'width': batsman_window[2], 'height': batsman_window[3]}
                    batsman_positions.append(batsman_info)
                
                # Display progress
                cv2.imshow('Cricket Analysis', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            frame_count += 1
            
        cap.release()
        cv2.destroyAllWindows()
        
        # Step 4: Map 2D to 3D coordinates
        ball_3d_positions = self.coordinator_3d.map_to_3d(ball_positions)
        
        # Step 5: Apply regression for trajectory smoothing
        smoothed_positions = self.coordinator_3d.apply_regression(ball_3d_positions)
        
        # Step 6: Make umpiring decisions
        batsman_height = self.estimate_batsman_height(batsman_positions)
        decisions = self.decision_maker.make_decisions(smoothed_positions, batsman_height)
        
        # Step 7: Visualize results
        self.visualizer.visualize(smoothed_positions, decisions)
        
        return {
            'ball_positions': ball_positions,
            'ball_3d_positions': ball_3d_positions,
            'smoothed_positions': smoothed_positions,
            'decisions': decisions
        }
    
    def estimate_batsman_height(self, batsman_positions):
        """Estimate batsman height from detected positions"""
        if not batsman_positions:
            return 1.75  # Default height in meters
            
        # Take average of detected heights
        heights = [pos['height'] for pos in batsman_positions]
        return np.mean(heights) * 0.01  # Convert to meters
        

class BallDetector:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.svm = None
        self.min_wdw_sz = (32, 32)
        self.step_size = 4
        self.scale = 1.05
        self.last_detection = None
        
        # Load or train SVM model
        self.load_or_train_model()
        
    def load_or_train_model(self):
        """Load existing SVM model or train a new one if not available"""
        model_path = "ball_svm_model.pkl"
        
        if os.path.exists(model_path):
            self.svm = cv2.ml.SVM_load(model_path)
            print("SVM model loaded from file")
        else:
            print("SVM model not found. Would need to train model with positive/negative samples.")
            # In a real implementation, we would train the model here
            # For this example, we'll use a placeholder model
            self.svm = SVC(kernel='linear', C=1.0)
            
    def extract_hog_features(self, img):
        """Extract HOG features from image"""
        # Resize image to match window size
        img_resized = cv2.resize(img, self.min_wdw_sz)
        
        # Convert to grayscale if needed
        if len(img_resized.shape) > 2:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # Calculate HOG features
        winSize = self.min_wdw_sz
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        features = hog.compute(img_resized)
        return features
        
    def sliding_window(self, image):
        """Implement sliding window to detect ball"""
        # If we have a previous detection, focus on that area
        if self.last_detection is not None:
            x, y, w, h = self.last_detection
            # Expand the window by 50%
            x = max(0, x - w//2)
            y = max(0, y - h//2)
            w = min(image.shape[1] - x, w * 2)
            h = min(image.shape[0] - y, h * 2)
            
            search_area = image[y:y+h, x:x+w]
            window_dims = (x, y, w, h)
        else:
            search_area = image
            window_dims = (0, 0, image.shape[1], image.shape[0])
            
        # Apply frame subtraction to focus on moving objects
        if hasattr(self, 'prev_frame') and self.prev_frame is not None:
            # Ensure prev_frame matches the search area size
            if self.prev_frame.shape != search_area.shape:
                self.prev_frame = cv2.resize(self.prev_frame, (search_area.shape[1], search_area.shape[0]))
                
            diff = cv2.absdiff(search_area, self.prev_frame)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)
            
            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size and shape
            for contour in contours:
                # Filter by size
                if cv2.contourArea(contour) < 50 or cv2.contourArea(contour) > 500:
                    continue
                    
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * cv2.contourArea(contour) / (perimeter * perimeter)
                if circularity < 0.5:  # Ball should be fairly circular
                    continue
                    
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Adjust coordinates to the original image
                x += window_dims[0]
                y += window_dims[1]
                
                # Extract window and check with HOG+SVM
                window = image[y:y+h, x:x+w]
                if window.size == 0:
                    continue
                    
                features = self.extract_hog_features(window)
                # In a real implementation, we would use:
                # if self.svm.predict(features.reshape(1, -1))[0] == 1:
                
                # For this example, we'll use a simplified approach
                # This would be replaced with actual SVM prediction
                if w/h > 0.8 and w/h < 1.2:  # Simple circularity check
                    self.last_detection = (x, y, w, h)
                    self.prev_frame = search_area.copy()
                    return (x, y, w, h)
        
        self.prev_frame = search_area.copy()
        return None
        
    def detect(self, frame):
        """Detect cricket ball in the frame"""
        return self.sliding_window(frame)


class BatsmanDetector:
    def __init__(self):
        # Use OpenCV's built-in people detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.last_detection = None
        
    def detect(self, frame):
        """Detect batsman in the frame"""
        # If we have a previous detection, focus on that area
        if self.last_detection is not None:
            x, y, w, h = self.last_detection
            # Expand the window a bit
            x = max(0, x - 20)
            y = max(0, y - 20)
            w = min(frame.shape[1] - x, w + 40)
            h = min(frame.shape[0] - y, h + 40)
            
            search_area = frame[y:y+h, x:x+w]
            boxes, weights = self.hog.detectMultiScale(search_area, winStride=(8, 8),
                                                      padding=(16, 16), scale=1.05)
            
            if len(boxes) > 0:
                # Apply non-maximum suppression
                if len(boxes) > 1:
                    indices = np.argsort(weights.flatten())[::-1]
                    boxes = boxes[indices]
                    
                # Get the highest-weighted detection
                best_box = boxes[0]
                
                # Adjust coordinates to the original frame
                adjusted_box = (
                    x + best_box[0],
                    y + best_box[1],
                    best_box[2],
                    best_box[3]
                )
                
                self.last_detection = adjusted_box
                return adjusted_box
        else:
            # Full frame detection
            boxes, weights = self.hog.detectMultiScale(frame, winStride=(8, 8),
                                                     padding=(16, 16), scale=1.05)
            
            if len(boxes) > 0:
                # Apply non-maximum suppression
                if len(boxes) > 1:
                    indices = np.argsort(weights.flatten())[::-1]
                    boxes = boxes[indices]
                    
                # Get the highest-weighted detection
                best_box = boxes[0]
                self.last_detection = tuple(best_box)
                return tuple(best_box)
                
        return None


class BallTracker:
    def __init__(self):
        self.prev_points = []
        self.max_distance = 50  # Maximum allowed distance between consecutive detections
        self.max_angle_diff = 45  # Maximum allowed angle difference in degrees
        
    def find_ball_center_radius(self, frame, ball_window):
        """Find the precise center and radius of the ball"""
        x, y, w, h = ball_window
        
        # Crop the ball window
        ball_crop = frame[y:y+h, x:x+w]
        if ball_crop.size == 0:
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(ball_crop, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # Find average brightness around center
        center_y, center_x = h // 2, w // 2
        center_region = blurred[
            max(0, center_y - 3):min(h, center_y + 4),
            max(0, center_x - 3):min(w, center_x + 4)
        ]
        
        avg_brightness = np.mean(center_region)
        
        # Threshold based on average brightness
        if avg_brightness > 128:
            _, thresholded = cv2.threshold(blurred, avg_brightness - 30, 255, cv2.THRESH_BINARY_INV)
        else:
            _, thresholded = cv2.threshold(blurred, avg_brightness + 30, 255, cv2.THRESH_BINARY)
            
        # Find contours
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Find the contour closest to the center with reasonable size
        best_contour = None
        min_dist = float('inf')
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10:  # Skip very small contours
                continue
                
            # Find center of contour
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
                
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Calculate distance to center of window
            dist = math.sqrt((cx - w/2)**2 + (cy - h/2)**2)
            
            if dist < min_dist:
                min_dist = dist
                best_contour = contour
                
        if best_contour is None:
            return None
            
        # Find minimum enclosing circle
        (cx, cy), radius = cv2.minEnclosingCircle(best_contour)
        
        # Adjust coordinates to original frame
        cx += x
        cy += y
        
        return int(cx), int(cy), int(radius)
        
    def is_valid_detection(self, new_point):
        """Check if the new detection is valid based on trajectory"""
        if len(self.prev_points) < 2:
            return True
            
        # Check distance from last point
        last_x, last_y, _ = self.prev_points[-1]
        dist = math.sqrt((new_point[0] - last_x)**2 + (new_point[1] - last_y)**2)
        
        if dist > self.max_distance:
            return False
            
        # Check angle if we have at least 2 previous points
        if len(self.prev_points) >= 2:
            prev_x, prev_y, _ = self.prev_points[-2]
            
            # Calculate angles
            angle1 = math.degrees(math.atan2(last_y - prev_y, last_x - prev_x))
            angle2 = math.degrees(math.atan2(new_point[1] - last_y, new_point[0] - last_x))
            
            # Calculate absolute difference between angles
            angle_diff = abs(angle1 - angle2)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
                
            if angle_diff > self.max_angle_diff:
                return False
                
        return True
        
    def detect_bounce(self):
        """Detect the bounce point in the trajectory"""
        if len(self.prev_points) < 3:
            return None
            
        # Find the lowest point in the trajectory
        min_y = float('inf')
        min_idx = -1
        
        for i, (_, y, _) in enumerate(self.prev_points):
            if y < min_y:
                min_y = y
                min_idx = i
                
        # Check if it's actually a bounce
        if min_idx > 0 and min_idx < len(self.prev_points) - 1:
            prev_y = self.prev_points[min_idx - 1][1]
            next_y = self.prev_points[min_idx + 1][1]
            
            if prev_y > min_y and next_y > min_y:
                return self.prev_points[min_idx]
                
        return None
        
    def track(self, frame, ball_window):
        """Track the ball and maintain its trajectory"""
        ball_info = self.find_ball_center_radius(frame, ball_window)
        
        if ball_info:
            x, y, radius = ball_info
            
            # Check if this detection is valid
            if self.is_valid_detection((x, y, radius)):
                self.prev_points.append((x, y, radius))
                
                # Detect if ball hit bat or batsman
                if len(self.prev_points) >= 3:
                    last_points = self.prev_points[-3:]
                    # Check for sudden change in direction
                    if len(last_points) == 3:
                        (x1, y1, _), (x2, y2, _), (x3, y3, _) = last_points
                        
                        # Calculate angles
                        angle1 = math.degrees(math.atan2(y2 - y1, x2 - x1))
                        angle2 = math.degrees(math.atan2(y3 - y2, x3 - x2))
                        
                        # Calculate absolute difference between angles
                        angle_diff = abs(angle1 - angle2)
                        if angle_diff > 180:
                            angle_diff = 360 - angle_diff
                            
                        if angle_diff > 90:  # Significant change in direction
                            # This could indicate ball hitting bat or batsman
                            return {'x': x, 'y': y, 'radius': radius, 'hit_detected': True}
                
                return {'x': x, 'y': y, 'radius': radius, 'hit_detected': False}
        
        return None


class Coordinator3D:
    def __init__(self):
        # Cricket pitch dimensions (in meters)
        self.pitch_length = 20.12  # Full pitch length
        self.pitch_width = 3.05
        self.wicket_height = 0.71
        
        # Calibration constants
        self.ball_radius_real = 0.036  # Cricket ball radius in meters
        self.ball_radius_at_crease = 10  # Pixel radius at crease (hypothetical)
        self.ball_radius_at_bowler = 5  # Pixel radius at bowler's end (hypothetical)
        
    def map_to_3d(self, ball_positions):
        """Map 2D image coordinates to 3D world coordinates"""
        if not ball_positions:
            return []
            
        result = []
        for pos in ball_positions:
            if not pos:  # Skip None values
                continue
                
            x, y, radius = pos['x'], pos['y'], pos['radius']
            
            # Scale x and y to world coordinates
            # This is a simplified approach - in reality, perspective correction would be needed
            x_world = (x / 640) * self.pitch_width - self.pitch_width/2
            
            # Calculate z (depth) from radius using inverse proportion
            radius_diff = self.ball_radius_at_crease - self.ball_radius_at_bowler
            z_world = (1 - (radius - self.ball_radius_at_bowler) / radius_diff) * self.pitch_length
            
            # Map y to height (simplified, not accounting for camera angle)
            y_world = (1 - y / 480) * 2  # Arbitrary height scaling
            
            result.append({'x': x_world, 'y': y_world, 'z': z_world, 'radius': radius})
            
        return result
        
    def apply_regression(self, positions_3d):
        """Apply regression to smooth out the trajectory"""
        if len(positions_3d) < 3:
            return positions_3d
            
        # Extract coordinates
        xs = [p['x'] for p in positions_3d]
        ys = [p['y'] for p in positions_3d]
        zs = [p['z'] for p in positions_3d]
        
        # Time points (assuming equal intervals)
        t = np.arange(len(xs))
        
        # Fit polynomial regression
        x_model = np.polyfit(t, xs, 2)
        y_model = np.polyfit(t, ys, 2)
        z_model = np.polyfit(t, zs, 1)  # Linear for z direction
        
        # Create polynomial functions
        x_poly = np.poly1d(x_model)
        y_poly = np.poly1d(y_model)
        z_poly = np.poly1d(z_model)
        
        # Apply smoothing
        smoothed = []
        for i, pos in enumerate(positions_3d):
            smoothed.append({
                'x': x_poly(i),
                'y': y_poly(i),
                'z': z_poly(i),
                'radius': pos['radius'],
                'original': pos
            })
            
        # Predict a few more points for visualization
        for i in range(len(positions_3d), len(positions_3d) + 5):
            smoothed.append({
                'x': x_poly(i),
                'y': y_poly(i),
                'z': z_poly(i),
                'radius': positions_3d[-1]['radius'] * 0.9,  # Approximate
                'predicted': True
            })
            
        return smoothed


class DecisionMaker:
    def __init__(self):
        # Cricket rule constants
        self.wide_margin = 0.75  # Distance in meters beyond which a ball is wide
        self.wicket_width = 0.23
        self.wicket_height = 0.71
        self.crease_length = 1.22
        
    def find_bounce_point(self, positions):
        """Find where the ball bounced on the pitch"""
        if len(positions) < 3:
            return None
            
        # Find local minimum in y-coordinate (height)
        for i in range(1, len(positions) - 1):
            prev_y = positions[i-1]['y'] 
            curr_y = positions[i]['y']
            next_y = positions[i+1]['y']
            
            if curr_y < prev_y and curr_y < next_y:
                return positions[i]
                
        return None
    
    def get_position_at_crease(self, positions):
        """Get ball position when it crosses the batsman's crease"""
        # Find position with z-coordinate closest to batsman's crease
        batsman_crease_z = 1.22  # Distance from wicket in meters
        
        closest_pos = None
        min_dist = float('inf')
        
        for pos in positions:
            dist = abs(pos['z'] - batsman_crease_z)
            if dist < min_dist:
                min_dist = dist
                closest_pos = pos
                
        return closest_pos
        
    def check_lbw(self, positions, batsman_height):
        """Check for Leg Before Wicket"""
        bounce_point = self.find_bounce_point(positions)
        if not bounce_point:
            return {'decision': 'Not Out', 'reason': 'No bounce detected'}
            
        # Find position when ball would hit wicket
        wicket_pos = None
        for pos in positions:
            if pos.get('predicted', False) and abs(pos['z']) < 0.5:  # Close to wicket
                wicket_pos = pos
                break
                
        if not wicket_pos:
            return {'decision': 'Not Out', 'reason': 'Ball would not hit wicket'}
            
        # Check if ball is in line with wicket
        if abs(wicket_pos['x']) > self.wicket_width/2:
            return {'decision': 'Not Out', 'reason': 'Ball would miss wicket'}
            
        # Check height at wicket
        if wicket_pos['y'] > self.wicket_height:
            return {'decision': 'Not Out', 'reason': 'Ball would pass over stumps'}
            
        # Check impact with batsman
        crease_pos = self.get_position_at_crease(positions)
        if not crease_pos:
            return {'decision': 'Not Out', 'reason': 'Cannot determine impact point'}
            
        # Simplified impact check - in reality would need batsman position
        if abs(crease_pos['x']) > 0.5:  # Arbitrary threshold for leg side
            return {'decision': 'Out', 'reason': 'LBW - Impact in line, would hit wicket'}
        else:
            return {'decision': 'Not Out', 'reason': 'Impact outside line of stumps'}
            
    def check_wide(self, positions, batsman_height):
        """Check for Wide ball"""
        crease_pos = self.get_position_at_crease(positions)
        if not crease_pos:
            return {'decision': 'Not Wide', 'reason': 'Cannot determine position at crease'}
            
        # Check off-side wide
        if crease_pos['x'] > self.wide_margin:
            return {'decision': 'Wide', 'reason': 'Ball passed too far outside off stump'}
            
        # Check leg-side wide (simplified)
        if crease_pos['x'] < -0.3:  # Arbitrary threshold for leg side
            return {'decision': 'Wide', 'reason': 'Ball passed down leg side'}
            
        # Check height
        if crease_pos['y'] > batsman_height:
            return {'decision': 'Wide', 'reason': 'Ball too high'}
            
        return {'decision': 'Not Wide', 'reason': 'Ball within playing area'}
        
    def check_no_ball(self, positions, batsman_height):
        """Check for No Ball (full toss above waist height)"""
        bounce_point = self.find_bounce_point(positions)
        crease_pos = self.get_position_at_crease(positions)
        
        if not crease_pos:
            return {'decision': 'Not No Ball', 'reason': 'Cannot determine position at crease'}
            
        # Check if ball didn't bounce before reaching batsman
        if not bounce_point or bounce_point['z'] > crease_pos['z']:
            # Check if height is above waist
            waist_height = batsman_height * 0.6  # Approximate waist height
            
            if crease_pos['y'] > waist_height:
                return {'decision': 'No Ball', 'reason': 'Full toss above waist height'}
                
        return {'decision': 'Not No Ball', 'reason': 'Legal delivery'}
        
    def check_bouncer(self, positions, batsman_height):
        """Check for Bouncer (ball bouncing over shoulder height)"""
        bounce_point = self.find_bounce_point(positions)
        crease_pos = self.get_position_at_crease(positions)
        
        if not bounce_point or not crease_pos:
            return {'decision': 'Not Bouncer', 'reason': 'Cannot determine bounce or position'}
            
        # Check if ball bounces before reaching batsman
        if bounce_point['z'] < crease_pos['z']:
            # Check if height is above shoulder
            shoulder_height = batsman_height * 0.8  # Approximate shoulder height
            
            if crease_pos['y'] > shoulder_height:
                return {'decision': 'Bouncer', 'reason': 'Ball bounced over shoulder height'}
                
        return {'decision': 'Not Bouncer', 'reason': 'Legal delivery'}
        
    def make_decisions(self, positions, batsman_height):
        """Make all umpiring decisions"""
        if not positions:
            return {'error': 'No ball tracking data available'}
            
        decisions = {
            'lbw': self.check_lbw(positions, batsman_height),
            'wide': self.check_wide(positions, batsman_height),
            'no_ball': self.check_no_ball(positions, batsman_height),
            'bouncer': self.check_bouncer(positions, batsman_height)
        }
        
        # Calculate ball speed
        if len(positions) >= 2:
            start_pos = positions[0]
            end_pos = positions[-1]
            distance = math.sqrt(
                (end_pos['x'] - start_pos['x'])**2 + 
                (end_pos['y'] - start_pos['y'])**2 + 
                (end_pos['z'] - start_pos['z'])**2
            )
            # Assuming frames are 1/30 sec apart
            time = len(positions) / 30  # seconds
            speed = distance / time  # m/s
            speed_kph = speed * 3.6  # Convert to km/h
            
            decisions['ball_speed'] = {
                'speed_ms': speed,
                'speed_kph': speed_kph
            }
        
        return decisions


class Visualizer3D:
    def __init__(self):
        self.scene = None
        
    def visualize(self, ball_positions, decisions):
        """Create 3D visualization of cricket pitch and ball trajectory"""
        # Create scene
        self.scene = vp.canvas(title="Cricket Ball Tracking Visualization", 
                              width=800, height=600,
                              center=vp.vector(0, 1, 10), background=vp.color.white)
                              
        # Create cricket pitch
        pitch_length = 20.12
        pitch_width = 3.05
        
        # Draw pitch
        vp.box(pos=vp.vector(0, 0, pitch_length/2),
              size=vp.vector(pitch_width, 0.01, pitch_length),
              color=vp.color.green)
              
        # Draw creases
        vp.box(pos=vp.vector(0, 0.01, 1.22),
              size=vp.vector(pitch_width, 0.01, 0.1),
              color=vp.color.white)
              
        vp.box(pos=vp.vector(0, 0.01, pitch_length-1.22),
              size=vp.vector(pitch_width, 0.01, 0.1),
              color=vp.color.white)
              
        # Draw wickets
        wicket_height = 0.71
        wicket_width = 0.23
        
        # Batsman's wicket
        vp.box(pos=vp.vector(0, wicket_height/2, 0),
              size=vp.vector(wicket_width, wicket_height, 0.05),
              color=vp.color.brown)
              
        # Bowler's wicket
        vp.box(pos=vp.vector(0, wicket_height/2, pitch_length),
              size=vp.vector(wicket_width, wicket_height, 0.05),
              color=vp.color.brown)
              
        # Draw ball trajectory
        prev_pos = None
        for i, pos in enumerate(ball_positions):
            # Create ball
            if pos.get('predicted', False):
                ball_color = vp.color.red  # Predicted positions
            else:
                ball_color = vp.color.blue  # Actual tracked positions
                
            # Convert coordinates
            ball_pos = vp.vector(pos['x'], pos['y'], pos['z'])
            
            # Create sphere for ball
            ball = vp.sphere(pos=ball_pos, radius=0.05, color=ball_color)
            
            # Draw trajectory line
            if prev_pos:
                vp.cylinder(pos=prev_pos, axis=ball_pos-prev_pos, 
                           radius=0.01, color=vp.color.gray(0.7))
            
            prev_pos = ball_pos
            
        # Display decisions
        self.display_decisions(decisions)
        
    def display_decisions(self, decisions):
        """Display umpiring decisions in the scene"""
        text_pos = vp.vector(4, 2, 10)  # Position for text display
        
        # Create text objects for each decision
        vp.text(pos=text_pos, text=f"LBW: {decisions['lbw']['decision']}", 
               align='left', height=0.3, color=vp.color.black)
               
        vp.text(pos=vp.vector(text_pos.x, text_pos.y-0.5, text_pos.z),
               text=f"Wide: {decisions['wide']['decision']}", 
               align='left', height=0.3, color=vp.color.black)
               
        vp.text(pos=vp.vector(text_pos.x, text_pos.y-1.0, text_pos.z),
               text=f"No Ball: {decisions['no_ball']['decision']}", 
               align='left', height=0.3, color=vp.color.black)
               
        vp.text(pos=vp.vector(text_pos.x, text_pos.y-1.5, text_pos.z),
               text=f"Bouncer: {decisions['bouncer']['decision']}", 
               align='left', height=0.3, color=vp.color.black)
               
        # Display ball speed if available
        if 'ball_speed' in decisions:
            speed = decisions['ball_speed']['speed_kph']
            vp.text(pos=vp.vector(text_pos.x, text_pos.y-2.0, text_pos.z),
                   text=f"Ball Speed: {speed:.1f} km/h", 
                   align='left', height=0.3, color=vp.color.black)


# GUI implementation using Kivy
class CricketAnalysisGUI:
    def __init__(self):
        self.system = CricketUmpireSystem()
        
    def build_gui(self):
        """Build the GUI using Kivy"""
        try:
            from kivy.app import App
            from kivy.uix.boxlayout import BoxLayout
            from kivy.uix.button import Button
            from kivy.uix.label import Label
            from kivy.uix.spinner import Spinner
            from kivy.uix.togglebutton import ToggleButton
            from kivy.uix.filechooser import FileChooserListView
            from kivy.config import Config
            
            Config.set('graphics', 'width', '800')
            Config.set('graphics', 'height', '600')
            
            class CricketAnalysisApp(App):
                def __init__(self, cricket_system, **kwargs):
                    super().__init__(**kwargs)
                    self.cricket_system = cricket_system
                    self.selected_video = ""
                    
                def build(self):
                    # Main layout
                    main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
                    
                    # Title
                    title = Label(text="Cricket Umpire Assistance System", 
                                 font_size=24, size_hint=(1, 0.1))
                    main_layout.add_widget(title)
                    
                    # Content layout
                    content_layout = BoxLayout(orientation='horizontal', spacing=10)
                    
                    # Left panel for controls
                    left_panel = BoxLayout(orientation='vertical', 
                                          size_hint=(0.4, 1), spacing=10)
                    
                    # Video selector
                    video_section = BoxLayout(orientation='vertical', size_hint=(1, 0.6))
                    video_label = Label(text="Videos", font_size=18)
                    video_section.add_widget(video_label)
                    
                    self.file_chooser = FileChooserListView(
                        filters=['*.mp4', '*.avi', '*.mov'],
                        path='.'
                    )
                    video_section.add_widget(self.file_chooser)
                    
                    # Selected video display
                    self.selected_video_label = Label(
                        text="Selected Video: None",
                        size_hint=(1, 0.1)
                    )
                    video_section.add_widget(self.selected_video_label)
                    
                    left_panel.add_widget(video_section)
                    
                    # Controls section
                    controls_section = BoxLayout(orientation='vertical', 
                                               size_hint=(1, 0.4), spacing=5)
                    
                    # Bowling action dropdown
                    action_layout = BoxLayout(orientation='horizontal')
                    action_label = Label(text="Select Bowling Action:", size_hint=(0.5, 1))
                    self.action_spinner = Spinner(
                        text='fast',
                        values=('fast', 'slow', 'no-action'),
                        size_hint=(0.5, 1)
                    )
                    action_layout.add_widget(action_label)
                    action_layout.add_widget(self.action_spinner)
                    controls_section.add_widget(action_layout)
                    
                    # Sliding windows toggle
                    toggle_layout = BoxLayout(orientation='horizontal')
                    toggle_label = Label(text="Show Sliding Windows:", size_hint=(0.5, 1))
                    self.toggle_button = ToggleButton(text='Off', size_hint=(0.5, 1))
                    toggle_layout.add_widget(toggle_label)
                    toggle_layout.add_widget(self.toggle_button)
                    controls_section.add_widget(toggle_layout)
                    
                    # Buttons for actions
                    button_layout = BoxLayout(orientation='vertical', spacing=5)
                    
                    self.play_button = Button(text="Play Video", size_hint=(1, 0.3))
                    self.play_button.bind(on_release=self.play_video)
                    
                    self.analyze_button = Button(text="Analyze", size_hint=(1, 0.3))
                    self.analyze_button.bind(on_release=self.analyze_video)
                    
                    self.visualize_button = Button(text="Visualize", size_hint=(1, 0.3))
                    self.visualize_button.bind(on_release=self.visualize_results)
                    
                    button_layout.add_widget(self.play_button)
                    button_layout.add_widget(self.analyze_button)
                    button_layout.add_widget(self.visualize_button)
                    
                    controls_section.add_widget(button_layout)
                    left_panel.add_widget(controls_section)
                    
                    # Right panel for output/display
                    right_panel = BoxLayout(orientation='vertical', size_hint=(0.6, 1))
                    
                    # Status display
                    self.status_label = Label(
                        text="Ready to analyze cricket video",
                        size_hint=(1, 0.1)
                    )
                    right_panel.add_widget(self.status_label)
                    
                    # Output display
                    self.output_label = Label(
                        text="No results yet",
                        size_hint=(1, 0.9),
                        text_size=(None, None),
                        halign='left',
                        valign='top'
                    )
                    right_panel.add_widget(self.output_label)
                    
                    # Add panels to content layout
                    content_layout.add_widget(left_panel)
                    content_layout.add_widget(right_panel)
                    
                    # Add content to main layout
                    main_layout.add_widget(content_layout)
                    
                    # File chooser event binding
                    self.file_chooser.bind(selection=self.select_video)
                    
                    return main_layout
                    
                def select_video(self, instance, selection):
                    """Handle video selection"""
                    if selection:
                        self.selected_video = selection[0]
                        self.selected_video_label.text = f"Selected Video: {self.selected_video.split('/')[-1]}"
                    
                def play_video(self, instance):
                    """Play the selected video"""
                    if not self.selected_video:
                        self.status_label.text = "Error: No video selected"
                        return
                        
                    # Open video with default player
                    import subprocess
                    import platform
                    
                    if platform.system() == "Windows":
                        os.startfile(self.selected_video)
                    elif platform.system() == "Darwin":  # macOS
                        subprocess.call(('open', self.selected_video))
                    else:  # Linux
                        subprocess.call(('xdg-open', self.selected_video))
                        
                    self.status_label.text = f"Playing video: {self.selected_video.split('/')[-1]}"
                    
                def analyze_video(self, instance):
                    """Analyze the selected video"""
                    if not self.selected_video:
                        self.status_label.text = "Error: No video selected"
                        return
                        
                    self.status_label.text = f"Analyzing video: {self.selected_video.split('/')[-1]}"
                    
                    # Run analysis in a separate thread to avoid blocking the UI
                    import threading
                    
                    def analysis_thread():
                        action_type = self.action_spinner.text
                        show_windows = self.toggle_button.state == 'down'
                        
                        try:
                            # Store original values
                            original_show_windows = self.cricket_system.ball_detector.show_windows
                            
                            # Set new values
                            self.cricket_system.ball_detector.show_windows = show_windows
                            
                            # Run analysis
                            results = self.cricket_system.process_video(
                                self.selected_video, 
                                action_type=action_type
                            )
                            
                            # Restore original values
                            self.cricket_system.ball_detector.show_windows = original_show_windows
                            
                            # Update UI with results
                            if results and 'decisions' in results:
                                decisions = results['decisions']
                                
                                output_text = (
                                    f"Analysis Results:\n\n"
                                    f"LBW: {decisions['lbw']['decision']}\n"
                                    f"Reason: {decisions['lbw']['reason']}\n\n"
                                    f"Wide: {decisions['wide']['decision']}\n"
                                    f"Reason: {decisions['wide']['reason']}\n\n"
                                    f"No Ball: {decisions['no_ball']['decision']}\n"
                                    f"Reason: {decisions['no_ball']['reason']}\n\n"
                                    f"Bouncer: {decisions['bouncer']['decision']}\n"
                                    f"Reason: {decisions['bouncer']['reason']}\n"
                                )
                                
                                if 'ball_speed' in decisions:
                                    output_text += f"\nBall Speed: {decisions['ball_speed']['speed_kph']:.1f} km/h"
                                    
                                # Update UI from main thread
                                from kivy.clock import Clock
                                Clock.schedule_once(
                                    lambda dt: self.update_results(output_text, "Analysis complete"), 0
                                )
                            else:
                                from kivy.clock import Clock
                                Clock.schedule_once(
                                    lambda dt: self.update_results("No valid results obtained", "Analysis failed"), 0
                                )
                        except Exception as e:
                            from kivy.clock import Clock
                            Clock.schedule_once(
                                lambda dt: self.update_results(f"Error: {str(e)}", "Analysis failed"), 0
                            )
                    
                    # Start analysis thread
                    threading.Thread(target=analysis_thread).start()
                    
                def update_results(self, output_text, status_text):
                    """Update UI with analysis results"""
                    self.output_label.text = output_text
                    self.status_label.text = status_text
                    
                def visualize_results(self, instance):
                    """Visualize the analysis results"""
                    if not hasattr(self.cricket_system, 'last_results') or not self.cricket_system.last_results:
                        self.status_label.text = "Error: No analysis results to visualize"
                        return
                        
                    self.status_label.text = "Opening visualization..."
                    
                    # Run visualization in a separate thread
                    import threading
                    
                    def vis_thread():
                        try:
                            positions = self.cricket_system.last_results['smoothed_positions']
                            decisions = self.cricket_system.last_results['decisions']
                            
                            self.cricket_system.visualizer.visualize(positions, decisions)
                            
                            from kivy.clock import Clock
                            Clock.schedule_once(
                                lambda dt: setattr(self.status_label, 'text', "Visualization complete"), 0
                            )
                        except Exception as e:
                            from kivy.clock import Clock
                            Clock.schedule_once(
                                lambda dt: setattr(self.status_label, 'text', f"Visualization error: {str(e)}"), 0
                            )
                    
                    # Start visualization thread
                    threading.Thread(target=vis_thread).start()
            
            # Create and run app
            app = CricketAnalysisApp(self.system)
            app.run()
            
        except ImportError as e:
            print(f"Error: Kivy library not found. {str(e)}")
            print("Please install Kivy using: pip install kivy")


# Main execution function
def main():
    """Main function to run the cricket umpire assistance system"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cricket Umpire Assistance System')
    parser.add_argument('--video', type=str, help='Path to cricket video file')
    parser.add_argument('--action', type=str, default='fast', 
                       choices=['fast', 'slow', 'no-action'],
                       help='Type of bowling action')
    parser.add_argument('--gui', action='store_true', help='Launch GUI interface')
    parser.add_argument('--visualize-only', action='store_true', 
                       help='Only visualize previously analyzed results')
    
    args = parser.parse_args()
    
    system = CricketUmpireSystem()
    
    if args.gui:
        # Launch GUI
        gui = CricketAnalysisGUI()
        gui.build_gui()
    elif args.visualize_only:
        # Load previous results and visualize
        try:
            import pickle
            with open('cricket_analysis_results.pkl', 'rb') as f:
                results = pickle.load(f)
                
            system.visualizer.visualize(
                results['smoothed_positions'], 
                results['decisions']
            )
        except FileNotFoundError:
            print("Error: No previous analysis results found")
    elif args.video:
        # Process video file
        results = system.process_video(args.video, action_type=args.action)
        
        if results:
            print("\nAnalysis Results:")
            decisions = results['decisions']
            
            print(f"\nLBW: {decisions['lbw']['decision']}")
            print(f"Reason: {decisions['lbw']['reason']}")
            
            print(f"\nWide: {decisions['wide']['decision']}")
            print(f"Reason: {decisions['wide']['reason']}")
            
            print(f"\nNo Ball: {decisions['no_ball']['decision']}")
            print(f"Reason: {decisions['no_ball']['reason']}")
            
            print(f"\nBouncer: {decisions['bouncer']['decision']}")
            print(f"Reason: {decisions['bouncer']['reason']}")
            
            if 'ball_speed' in decisions:
                print(f"\nBall Speed: {decisions['ball_speed']['speed_kph']:.1f} km/h")
                
            # Save results for later visualization
            import pickle
            with open('cricket_analysis_results.pkl', 'wb') as f:
                pickle.dump(results, f)
                
            # Show visualization
            system.visualizer.visualize(
                results['smoothed_positions'], 
                results['decisions']
            )
        else:
            print("Analysis failed to produce valid results")
    else:
        print("Please provide a video file with --video or use --gui to launch the GUI")
        parser.print_help()


if __name__ == "__main__":
    main()