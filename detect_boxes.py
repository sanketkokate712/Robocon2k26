import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
import os
import sys

class BoxDetector(Node):
    def __init__(self):
        super().__init__('box_detector')
        self.subscription = self.create_subscription(
            Image,
            '/camera1/image_raw',
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning
        
        # Load reference images
        self.base_path = '/home/sanket/robocon/models'
        self.references = {
            "Robocon": os.path.join(self.base_path, 'box_robocon/materials/textures/robocon_logo.png'),
            "D-01": os.path.join(self.base_path, 'box_d01/materials/textures/d01_logo.png'),
            "Fake-KFS": os.path.join(self.base_path, 'box_custom/materials/textures/fake_kfs.png')
        }
        
        self.descriptors = {}
        self.keypoints = {}
        self.orb = cv2.ORB_create(nfeatures=1000)
        
        # Pre-compute features for reference images
        print("Loading reference images...")
        for name, path in self.references.items():
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    kp, des = self.orb.detectAndCompute(img, None)
                    if des is not None:
                        self.keypoints[name] = kp
                        self.descriptors[name] = des
                        print(f"Loaded {name}: {len(kp)} features")
                    else:
                        print(f"Warning: No features found in {name}")
                else:
                    print(f"Error: Could not read image {path}")
            else:
                print(f"Error: File not found {path}")

        # Matcher
        # Matcher
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV Image manually (to avoid cv_bridge dependency issues)
        try:
            # Assuming 'rgb8' or 'bgr8' encoding based on Gazebo plugin R8G8B8
            # Gazebo R8G8B8 usually maps to rgb8
            dtype = np.uint8
            n_channels = 3
            if msg.encoding == 'mono8':
                n_channels = 1
            
            img = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, n_channels)
            
            # Convert RGB to BGR for OpenCV if necessary
            if msg.encoding == 'rgb8':
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            self.process_image(img)
            
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')

    def process_image(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_scene, des_scene = self.orb.detectAndCompute(gray, None)
        
        if des_scene is None:
            cv2.imshow("Box Detection", frame)
            cv2.waitKey(1)
            return

        for name, des_ref in self.descriptors.items():
            if des_ref is None: continue
            
            # KNN Match with k=2
            try:
                matches = self.bf.knnMatch(des_ref, des_scene, k=2)
            except Exception:
                continue
            
            # Apply Ratio Test
            good = []
            for m, n in matches:
                # Stricter ratio test to avoid false matches across different boxes
                if m.distance < 0.7 * n.distance:
                    good.append(m)
            
            # Minimum matches required
            MIN_MATCH_COUNT = 10
            
            if len(good) > MIN_MATCH_COUNT:
                # print(f"Detected {name} with {len(good)} matches")
                src_pts = np.float32([self.keypoints[name][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                
                # Find Homography
                # Increased threshold to 5.0 to allow for some perspective distortion, but RANSAC should handle outliers
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if M is not None:
                    h, w = cv2.imread(self.references[name], cv2.IMREAD_GRAYSCALE).shape
                    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                    try:
                        dst = cv2.perspectiveTransform(pts, M)
                        
                        # Check if the detected polygon is convex and has reasonable area to filter noise
                        if cv2.isContourConvex(np.int32(dst)):
                            area = cv2.contourArea(np.int32(dst))
                            # Box should be substantial but not spanning the entire screen unreasonably
                            # 100 < area < 300000 (roughly half screen)
                            if area > 500 and area < 300000: 
                                frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                                
                                # Draw Label
                                x, y = np.int32(dst[0][0])
                                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            else:
                                pass
                                # print(f"Ignored {name} due to area: {area}")
                        else:
                            pass
                            # print(f"Ignored {name} - contour not convex")
                    except Exception as e:
                        pass # Transform failed

        cv2.imshow("Box Detection", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    box_detector = BoxDetector()
    print("Box Detector Node Started. Waiting for camera images...")
    try:
        rclpy.spin(box_detector)
    except KeyboardInterrupt:
        pass
    finally:
        box_detector.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
