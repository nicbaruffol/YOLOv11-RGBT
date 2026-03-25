import rosbag
from cv_bridge import CvBridge

bag = rosbag.Bag('/home/radar/catkin_ws/bags/flight_mavic_1/drone_mavic_flight_01.bag')
bridge = CvBridge()
for topic, msg, t in bag.read_messages(topics=['/blackfly_s/image_raw']):
    print(f"Width: {msg.width}, Height: {msg.height}")
    break # Only check the first frame
bag.close()
