#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
import subprocess
import signal
import os
from datetime import datetime
import threading


# Import PySpin for the Point Grey / FLIR industrial camera
try:
    import PySpin
except ImportError:
    print("⚠️ PySpin not found! Spinnaker SDK cameras will not work.")


# --- PASTE YOUR SpinnakerStream AND CameraStream CLASSES HERE ---
# (Copy them exactly as they were in the previous live_app.py script)

# --- THREADED V4L2 CAMERA CLASS (For the IR Boson) ---
class CameraStream:
    def __init__(self, src=0):
        try:
            src = int(src)
        except ValueError:
            pass 

        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            self.grabbed, self.frame, self.stopped = False, None, True
            return

        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        if not self.stopped:
            threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        if hasattr(self, 'stream') and self.stream.isOpened():
            self.stream.release()

# --- THREADED SPINNAKER CAMERA CLASS (For Point Grey RGB) ---
class SpinnakerStream:
    def __init__(self):
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()
        
        if self.cam_list.GetSize() == 0:
            print("⚠️ No Spinnaker cameras found!")
            self.cam_list.Clear()
            self.system.ReleaseInstance()
            self.grabbed, self.frame, self.stopped = False, None, True
            return

        # Grab the first available Spinnaker camera
        self.cam = self.cam_list.GetByIndex(0)
        self.cam.Init()
        
        # 1. HARDWARE LEVEL FIX: Force the camera to output BGR8 natively
        try:
            # If the camera node is writable, switch it from Raw/Bayer to BGR8
            if self.cam.PixelFormat.GetAccessMode() == PySpin.RW:
                self.cam.PixelFormat.SetValue(PySpin.PixelFormat_BGR8)
                print("✅ Spinnaker hardware successfully set to BGR8 mode.")
            else:
                print("⚠️ Spinnaker PixelFormat is Read-Only. Falling back to software conversion.")
        except Exception as e:
            print(f"⚠️ Could not set hardware BGR8 mode: {e}")

        # 2. SOFTWARE FALLBACK: Setup Image Processor (if it exists in your SDK version)
        self.processor = None
        if hasattr(PySpin, 'ImageProcessor'):
            self.processor = PySpin.ImageProcessor()
            try:
                self.processor.SetColorProcessing(PySpin.HQ_LINEAR)
            except AttributeError:
                pass
        
        # Set acquisition mode to continuous
        self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        self.cam.BeginAcquisition()

        self.grabbed = True
        self.frame = None
        self.stopped = False

    def start(self):
        if not self.stopped:
            threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            try:
                # Grab image from camera buffer (1000ms timeout)
                image_result = self.cam.GetNextImage(1000)
                
                if not image_result.IsIncomplete():
                    # Extract the raw NumPy array from the buffer
                    raw_array = image_result.GetNDArray()
                    
                    # Check if the hardware gave us a 3-channel color image natively
                    if len(raw_array.shape) == 3 and raw_array.shape[2] == 3:
                        self.frame = raw_array
                        
                    # If the hardware gave us a 2D Grayscale/Bayer image, we must convert it
                    else:
                        # Attempt SDK 4.x Processor Conversion
                        if self.processor is not None:
                            converted = self.processor.Convert(image_result, PySpin.PixelFormat_BGR8)
                            self.frame = converted.GetNDArray()
                            
                        # Attempt SDK 2.x Legacy Conversion
                        elif hasattr(image_result, 'Convert'):
                            converted = image_result.Convert(PySpin.PixelFormat_BGR8, PySpin.HQ_LINEAR)
                            self.frame = converted.GetNDArray()
                            
                        # ULTIMATE BULLETPROOF FALLBACK: OpenCV Conversion
                        # This guarantees the app will not crash so YOLO can run!
                        else:
                            self.frame = cv2.cvtColor(raw_array, cv2.COLOR_GRAY2BGR)
                            
                    self.grabbed = True
                
                # We MUST release the image to free the buffer
                image_result.Release()
                
            except PySpin.SpinnakerException as e:
                print(f"Spinnaker Stream Error: {e}")
                self.grabbed = False

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        try:
            if hasattr(self, 'cam') and self.cam.IsInitialized():
                self.cam.EndAcquisition()
                self.cam.DeInit()
                del self.cam
            if hasattr(self, 'cam_list'):
                self.cam_list.Clear()
            if hasattr(self, 'system'):
                self.system.ReleaseInstance()
        except Exception as e:
            print(f"Error shutting down Spinnaker: {e}")

# --- MAIN FUNCTION ---

def main():
    # 1. Initialize the ROS Node
    rospy.init_node('multimodal_camera_publisher', anonymous=True)
    
    # 2. Setup ROS Publishers and Bridge
    rgb_pub = rospy.Publisher('/camera/rgb/image_raw', Image, queue_size=10)
    ir_pub = rospy.Publisher('/camera/ir/image_raw', Image, queue_size=10)
    bridge = CvBridge()

    # 3. Initialize Cameras
    print("Initializing cameras...")
    rgb_stream = SpinnakerStream().start() # Or CameraStream(src=2) if testing
    ir_stream = CameraStream(src=0).start()
    
    time.sleep(1.0) # Hardware warmup
    
    recording = False
    rosbag_proc = None
    os.makedirs("rosbags", exist_ok=True)
    
    # --- UPDATE THIS TO YOUR ACTUAL RADAR TOPIC ---
    radar_topic = "/radar/data" 
    
    print("\n✅ ROS Camera Node Running!")
    print("Press [SPACEBAR] in the preview window to Start/Stop rosbag recording.")
    print("Press [ESC] or [Q] to quit.")

    try:
        while not rospy.is_shutdown():
            rgb_frame = rgb_stream.read()
            ir_frame = ir_stream.read()
            
            if rgb_frame is None or ir_frame is None:
                continue

            # --- 1. PUBLISH TO ROS ---
            # Grab a SINGLE timestamp for perfect software synchronization
            sync_time = rospy.Time.now()

            # Convert OpenCV frames (NumPy) to ROS Image messages
            rgb_msg = bridge.cv2_to_imgmsg(rgb_frame, encoding="bgr8")
            ir_msg = bridge.cv2_to_imgmsg(ir_frame, encoding="bgr8")
            
            # Apply the exact same timestamp to both headers
            rgb_msg.header.stamp = sync_time
            ir_msg.header.stamp = sync_time
            rgb_msg.header.frame_id = "camera_link"
            ir_msg.header.frame_id = "camera_link"

            # Publish the frames to the ROS network
            rgb_pub.publish(rgb_msg)
            ir_pub.publish(ir_msg)

            # --- 2. LIVE PREVIEW & UI ---
            display_rgb = rgb_frame.copy()
            
            if recording:
                # Draw a red recording dot
                cv2.circle(display_rgb, (30, 30), 10, (0, 0, 255), -1)

            ir_preview = cv2.resize(ir_frame, (display_rgb.shape[1], display_rgb.shape[0]))
            preview = cv2.hconcat([display_rgb, ir_preview])
            cv2.imshow("ROS Logger (Left: RGB, Right: IR)", cv2.resize(preview, (1280, 480)))

            # --- 3. KEYBOARD CONTROLS ---
            key = cv2.waitKey(1) & 0xFF
            
            # Start / Stop Recording (Spacebar)
            if key == ord(' '):
                recording = not recording
                
                if recording:
                    # START ROSBAG
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    bag_name = f"rosbags/multimodal_{timestamp}.bag"
                    print(f"\n🔴 STARTING ROSBAG: {bag_name}")
                    
                    # Launch rosbag record in the background via subprocess
                    command = [
                        "rosbag", "record", 
                        "-O", bag_name, 
                        "/camera/rgb/image_raw", 
                        "/camera/ir/image_raw", 
                        radar_topic  # Captures your Ethernet radar stream!
                    ]
                    rosbag_proc = subprocess.Popen(command)
                    
                else:
                    # STOP ROSBAG
                    print("\n⏹️ STOPPING ROSBAG. Saving file...")
                    if rosbag_proc:
                        # Send CTRL+C signal to gracefully stop rosbag and save the index
                        rosbag_proc.send_signal(signal.SIGINT)
                        rosbag_proc.wait()
                        rosbag_proc = None
                        print("✅ Rosbag saved successfully.")

            # Quit (Esc or Q)
            elif key == 27 or key == ord('q'):
                break

    finally:
        # Emergency cleanup if you close the terminal
        if rosbag_proc:
            rosbag_proc.send_signal(signal.SIGINT)
            rosbag_proc.wait()
        rgb_stream.stop()
        ir_stream.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()