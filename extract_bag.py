import rosbag
import cv2
import numpy as np

def extract_videos(bag_file, rgb_topic, ir_topic, rgb_out="rgb_fixed.mp4", ir_out="ir_fixed.mp4"):
    bag = rosbag.Bag(bag_file, 'r')

    print(f"Scanning {bag_file} to calculate exact real-world framerates...")

    # --- PASS 1: Calculate exact FPS ---
    stats = {
        rgb_topic: {'count': 0, 'start_time': None, 'end_time': None},
        ir_topic:  {'count': 0, 'start_time': None, 'end_time': None}
    }

    for topic, msg, t in bag.read_messages(topics=[rgb_topic, ir_topic]):
        msg_time = msg.header.stamp.to_sec() # Use the exact hardware timestamp
        
        if stats[topic]['start_time'] is None:
            stats[topic]['start_time'] = msg_time
            
        stats[topic]['end_time'] = msg_time
        stats[topic]['count'] += 1

    # Do the math
    rgb_duration = stats[rgb_topic]['end_time'] - stats[rgb_topic]['start_time']
    ir_duration = stats[ir_topic]['end_time'] - stats[ir_topic]['start_time']
    
    rgb_fps = stats[rgb_topic]['count'] / rgb_duration if rgb_duration > 0 else 30.0
    ir_fps = stats[ir_topic]['count'] / ir_duration if ir_duration > 0 else 9.0

    print(f"RGB Camera: {stats[rgb_topic]['count']} frames over {rgb_duration:.2f}s = {rgb_fps:.2f} FPS")
    print(f"IR Camera:  {stats[ir_topic]['count']} frames over {ir_duration:.2f}s = {ir_fps:.2f} FPS")
    print("Extracting videos now...\n")

    # --- PASS 2: Write the videos ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    rgb_writer = None
    ir_writer = None

    for topic, msg, t in bag.read_messages(topics=[rgb_topic, ir_topic]):
        # Convert ROS Image message to OpenCV format
        try:
            if topic == rgb_topic:
                cv_img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                if msg.encoding == "rgb8": 
                    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
            
            elif topic == ir_topic:
                encoding = msg.encoding
                
                # Standard 8-bit grayscale
                if encoding in ['mono8', '8UC1']:
                    cv_img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
                
                # 16-bit thermal data (The most likely culprit!)
                elif encoding in ['mono16', '16UC1']:
                    # Read as 16-bit integers
                    cv_img_16 = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
                    
                    # Normalize the temperature values to 0-255 so we can actually see it in an MP4
                    cv_img = cv2.normalize(cv_img_16, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                
                # False-color thermal (already processed by the camera into RGB/BGR)
                elif encoding in ['bgr8', 'rgb8', '8UC3']:
                    cv_img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                    if encoding == "rgb8":
                        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
                else:
                    print(f"Skipping unknown IR encoding: {encoding}")
                    continue
                    
        except Exception as e:
            print(f"Skipping frame due to decoding error: {e}")
            continue

        # If it's a 1-channel IR image, convert to 3-channel so VideoWriter doesn't complain
        if len(cv_img.shape) == 2:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)

        # Initialize writers on the very first frame to get exact width/height
        if topic == rgb_topic:
            if rgb_writer is None:
                h, w = cv_img.shape[:2]
                rgb_writer = cv2.VideoWriter(rgb_out, fourcc, rgb_fps, (w, h))
            rgb_writer.write(cv_img)
            
        elif topic == ir_topic:
            if ir_writer is None:
                h, w = cv_img.shape[:2]
                ir_writer = cv2.VideoWriter(ir_out, fourcc, ir_fps, (w, h))
            ir_writer.write(cv_img)

    # Cleanup
    if rgb_writer: rgb_writer.release()
    if ir_writer: ir_writer.release()
    bag.close()
    
    print(f"Done! Saved perfectly timed videos to '{rgb_out}' and '{ir_out}'.")

if __name__ == '__main__':
    # ⚠️ CHANGE THESE TO MATCH YOUR ACTUAL SETUP ⚠️
    BAG_FILE = "/home/radar/catkin_ws/bags/flight_mavic_2/drone_mavic_flight_02.bag"
    RGB_TOPIC = "/blackfly_s/image_raw"
    IR_TOPIC = "/flir_boson/image_raw" 
    
    extract_videos(BAG_FILE, RGB_TOPIC, IR_TOPIC)