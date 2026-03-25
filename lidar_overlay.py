import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
import yaml
import os # Make sure this is imported!

class RadarVisualizer:
    def __init__(self):
        self.latest_img = None
        self.latest_points = None
        
        # --- CONFIGURATION ---
        self.img_topic = "/blackfly_s/image_raw" 
        self.radar_topic = "/zadar/scan0"

        # 1. Create Window FIRST to avoid the Null Pointer Error
        cv2.namedWindow('Radar Alignment', cv2.WINDOW_NORMAL)

        # 2. Create Trackbars (Order: X, Y, Z, Yaw, Pitch, Focal)
        cv2.createTrackbar('X (cm)', 'Radar Alignment', 100, 200, lambda x: None)
        cv2.createTrackbar('Y (cm)', 'Radar Alignment', 100, 200, lambda x: None)
        cv2.createTrackbar('Z (cm)', 'Radar Alignment', 100, 200, lambda x: None)
        cv2.createTrackbar('Yaw', 'Radar Alignment', 180, 360, lambda x: None)
        cv2.createTrackbar('Pitch', 'Radar Alignment', 180, 360, lambda x: None)
        cv2.createTrackbar('Focal (Scale)', 'Radar Alignment', 1000, 2000, lambda x: None)

        # 3. Auto-Load Saved Calibration
        import os
        if os.path.exists("radar_calib.yaml"):
            with open("radar_calib.yaml", "r") as f:
                calib = yaml.safe_load(f)
                if calib:
                    print("\n[INFO] Found radar_calib.yaml! Loading saved settings...")
                    cv2.setTrackbarPos('X (cm)', 'Radar Alignment', calib.get('x_val', 100))
                    cv2.setTrackbarPos('Y (cm)', 'Radar Alignment', calib.get('y_val', 100))
                    cv2.setTrackbarPos('Z (cm)', 'Radar Alignment', calib.get('z_val', 100))
                    cv2.setTrackbarPos('Yaw', 'Radar Alignment', calib.get('yaw_val', 180))
                    cv2.setTrackbarPos('Pitch', 'Radar Alignment', calib.get('pitch_val', 180))
                    cv2.setTrackbarPos('Focal (Scale)', 'Radar Alignment', calib.get('f_val', 1000))

        # ROS Subscribers
        rospy.Subscriber(self.img_topic, Image, self.img_callback)
        rospy.Subscriber(self.radar_topic, PointCloud2, self.pc_callback)
        print(f"Listening to {self.radar_topic}. Press 'S' to save calibration.")

    def img_callback(self, msg):
        # Manual byte-to-image conversion
        img_array = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        self.latest_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR) if "rgb" in msg.encoding else img_array

    def pc_callback(self, msg):
        try:
            # Add "range" to the requested fields
            # *Note: ROS field names are case-sensitive. If this crashes, try "Range"*
            pts_gen = pc2.read_points(msg, field_names=("x", "y", "z", "range"), skip_nans=True)
            
            # The array is now Nx4: [x, y, z, range]
            self.latest_points = np.array(list(pts_gen), dtype=np.float32)
        except Exception as e:
            # Adding a print here so we know if the field name is slightly different
            print(f"Error reading points (check field name): {e}")

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            img_curr = self.latest_img
            pts_curr = self.latest_points

            if img_curr is not None:
                display = img_curr.copy()
                h, w = display.shape[:2]

                if pts_curr is not None and len(pts_curr) > 0:
                    # 1. Get FINE-TUNING values from sliders (180 = 0 offset)
                    tx = (cv2.getTrackbarPos('X (cm)', 'Radar Alignment') - 100) / 100.0
                    ty = (cv2.getTrackbarPos('Y (cm)', 'Radar Alignment') - 100) / 100.0
                    tz = (cv2.getTrackbarPos('Z (cm)', 'Radar Alignment') - 100) / 100.0
                    
                    # Small tweaks (Yaw/Pitch) around the zero-point
                    yaw_tweak = np.radians(cv2.getTrackbarPos('Yaw', 'Radar Alignment') - 180)
                    pitch_tweak = np.radians(cv2.getTrackbarPos('Pitch', 'Radar Alignment') - 180)
                    f_scale = cv2.getTrackbarPos('Focal (Scale)', 'Radar Alignment')

                    # 2. THE BREAKTHROUGH: Hardcode the axes swap!
                    # Radar (X=Fwd, Y=Left, Z=Up) -> Camera (Z=Fwd, X=Right, Y=Down)
                    # So: Cam_X = -Radar_Y | Cam_Y = -Radar_Z | Cam_Z = Radar_X
                    cam_pts = np.zeros_like(pts_curr[:, :3])
                    cam_pts[:, 0] = -pts_curr[:, 1]  # X becomes Right
                    cam_pts[:, 1] = -pts_curr[:, 2]  # Y becomes Down
                    cam_pts[:, 2] = pts_curr[:, 0]   # Z becomes Forward

                    # 3. Filter out points BEHIND the camera (Z <= 0)
                    # This stops the wild scattering!
                    front_mask = cam_pts[:, 2] > 0.1 
                    cam_pts = cam_pts[front_mask]
                    valid_ranges = pts_curr[front_mask, 3] # Keep matching ranges

                    if len(cam_pts) > 0:
                        # Build Projection Matrix
                        K = np.array([[f_scale, 0, w/2], [0, f_scale, h/2], [0, 0, 1]], dtype=np.float32)
                        dist_coeffs = np.zeros(5) 

                        # 1. Create true Euler Rotation Matrices
                        # Pitch is rotation around the Camera's X-axis (up/down)
                        Rx = np.array([
                            [1, 0, 0],
                            [0, np.cos(pitch_tweak), -np.sin(pitch_tweak)],
                            [0, np.sin(pitch_tweak), np.cos(pitch_tweak)]
                        ], dtype=np.float32)

                        # Yaw is rotation around the Camera's Y-axis (left/right)
                        Ry = np.array([
                            [np.cos(yaw_tweak), 0, np.sin(yaw_tweak)],
                            [0, 1, 0],
                            [-np.sin(yaw_tweak), 0, np.cos(yaw_tweak)]
                        ], dtype=np.float32)

                        # Combine the rotations (Ry * Rx)
                        R_combined = Ry @ Rx 
                        
                        # Convert the clean rotation matrix into OpenCV's Rodrigues format
                        rvec, _ = cv2.Rodrigues(R_combined)
                        
                        # Translation vector stays the same
                        tvec = np.array([tx, ty, tz], dtype=np.float32)

                        # Project 3D -> 2D 
                        spatial_pts = np.ascontiguousarray(cam_pts, dtype=np.float32)
                        img_pts, _ = cv2.projectPoints(spatial_pts, rvec, tvec, K, dist_coeffs)
                        img_pts = img_pts.reshape(-1, 2)

                        # --- HIGH SPEED VECTORIZATION ---
                        
                        # 1. Extract X and Y pixel coordinates for all points at once
                        u = img_pts[:, 0].astype(int)
                        v = img_pts[:, 1].astype(int)
                        
                        # 2. Create a single filter mask for screen boundaries AND valid ranges
                        valid_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (valid_ranges > 20.0)
                        
                        # Apply mask to keep only the points we actually want to draw
                        u_valid = u[valid_mask]
                        v_valid = v[valid_mask]
                        ranges_valid = valid_ranges[valid_mask]

                        if len(u_valid) > 0:
                            # 3. Calculate colors for ALL valid points simultaneously
                            norm_dist = np.clip((ranges_valid - 20.0) / 100.0, 0, 1)
                            intensities = np.uint8(255 * (1 - norm_dist))
                            
                            # applyColorMap can process an entire array at once!
                            colors = cv2.applyColorMap(intensities.reshape(-1, 1), cv2.COLORMAP_JET)
                            colors = colors.reshape(-1, 3) # Reshape back to Nx3 BGR list

                            # 4. Ultra-tight drawing loop
                            for i in range(len(u_valid)):
                                # cv2.circle requires native Python ints
                                c = (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2]))
                                cv2.circle(display, (u_valid[i], v_valid[i]), 3, c, -1)

                cv2.imshow('Radar Alignment', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # Save the raw slider integer values, NOT the numpy floats
                calib = {
                    "x_val": cv2.getTrackbarPos('X (cm)', 'Radar Alignment'),
                    "y_val": cv2.getTrackbarPos('Y (cm)', 'Radar Alignment'),
                    "z_val": cv2.getTrackbarPos('Z (cm)', 'Radar Alignment'),
                    "yaw_val": cv2.getTrackbarPos('Yaw', 'Radar Alignment'),
                    "pitch_val": cv2.getTrackbarPos('Pitch', 'Radar Alignment'),
                    "f_val": cv2.getTrackbarPos('Focal (Scale)', 'Radar Alignment')
                }
                with open("radar_calib.yaml", "w") as f_out:
                    yaml.dump(calib, f_out)
                print("\n[SUCCESS] Calibration saved safely to radar_calib.yaml!")

            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('radar_overlay')
    rv = RadarVisualizer()
    rv.run()