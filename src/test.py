#!usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
# Khởi tạo node ROS
rospy.init_node('image_processing_node', anonymous=True)

# Khởi tạo cv_bridge
bridge = CvBridge()
scan_list = []

def image_callback(msg):
    try:
        # Chuyển đổi hình ảnh ROS sang OpenCV
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        #scan =(cv_image * 255).astype(np.uint8)
        cv_img=np.nan_to_num(cv_img, nan=0.0)
        cv_img[cv_img < 0.4] = 0.
        cv_img/=(10./255.)
        cv_img = np.array(cv_img, dtype=np.float32)
        cv_img*=(10./255.)
        cv_img=(cv_img/5.)
        print(cv_img)

        cv2.imshow('sss',cv_img)
       
        
        cv2.waitKey(1)
    except CvBridgeError as e:
        rospy.logerr(e)

def main():
    # Đăng ký với topic hình ảnh bạn muốn theo dõi
    rospy.Subscriber("/kinect/depth/image_raw", Image, image_callback)

    # Khởi tạo cửa sổ OpenCV


    # Lặp vô hạn để giữ cửa sổ mở
    rospy.spin()

if __name__ == '__main__':
    main()
