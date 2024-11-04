#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge, CvBridgeError

Red = (0, 0, 255)
Blue = (255, 0, 0)
Green = (0, 255, 0)

WIDTH = 640
HEIGHT = 480

ROI_WIDTH = WIDTH
ROI_HEIGHT = HEIGHT // 2
BOTTOM_CENTER_X = WIDTH // 2
BOTTOM_CENTER_Y = HEIGHT - (ROI_HEIGHT // 2)

TOP_LEFT = (BOTTOM_CENTER_X - ROI_WIDTH // 2, HEIGHT - ROI_HEIGHT)
BOTTOM_RIGHT = (BOTTOM_CENTER_X + ROI_WIDTH // 2, HEIGHT)

last_avg_x = None
last_avg_y = None
last_distance = None 

def apply_roi(img, top_left, bottom_right):
    mask = np.zeros_like(img)
    cv2.rectangle(mask, top_left, bottom_right, (255, 255, 255), thickness=-1)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img

def adjust_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:, :, 2])
    if brightness < 100:  
        ratio = 130.0 / brightness
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * ratio, 0, 255)
        adjusted_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return adjusted_img
    return image

def lane_detect(image):
    global last_avg_x, last_avg_y, last_distance 

    image = adjust_brightness(image)
    image = enhance_contrast(image)

    img = image.copy()
    display_img = img.copy()  

    roi_img = apply_roi(img, TOP_LEFT, BOTTOM_RIGHT)

    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edge_img = cv2.Canny(np.uint8(blur_gray), 30, 170)
   
    edge_img_rgb = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2BGR)

    horizon_y = HEIGHT * 9 // 10
    cv2.line(edge_img_rgb, (0, horizon_y), (WIDTH, horizon_y), Blue, 2)

    nonzero = edge_img[horizon_y, :].nonzero()[0]
    left_points = nonzero[nonzero < WIDTH // 2]
    right_points = nonzero[nonzero >= WIDTH // 2]
    
    for x in nonzero:
        cv2.circle(edge_img_rgb, (x, horizon_y), 5, Red, -1)

    if len(left_points) > 0 and len(right_points) > 0:
        left_avg_x = int(np.mean(left_points))
        right_avg_x = int(np.mean(right_points))
        avg_x = (left_avg_x + right_avg_x) // 2
        avg_y = horizon_y
        last_distance = avg_x - left_avg_x 
        last_avg_x = avg_x
        last_avg_y = avg_y
        cv2.circle(edge_img_rgb, (avg_x, avg_y), 10, Green, -1)
        return display_img, edge_img_rgb, avg_x, avg_y

    elif len(left_points) > 0 or len(right_points) > 0:
        if last_avg_x is None and last_avg_y is None:
            return display_img, edge_img_rgb, None, None
        else:
            if len(left_points) > 0:
                left_avg_x = int(np.mean(left_points))
                avg_x = left_avg_x + (last_distance if last_distance is not None else 0)
            else:
                right_avg_x = int(np.mean(right_points))
                avg_x = right_avg_x - (last_distance if last_distance is not None else 0)
            avg_y = horizon_y
            last_avg_x = avg_x
            last_avg_y = avg_y
            cv2.circle(edge_img_rgb, (avg_x, avg_y), 10, Green, -1)
            return display_img, edge_img_rgb, avg_x, avg_y

    else:
        last_avg_x = None
        last_avg_y = None
        last_distance = None
        return display_img, edge_img_rgb, None, None

def image_callback(msg):
    global last_avg_x, last_avg_y

    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        rospy.logerr("CvBridgeError: {0}".format(e))
        return

    processed_frame, edge_img, avg_x, avg_y = lane_detect(cv_image)

    if avg_x is not None and avg_y is not None:
        center_point_msg = Int32MultiArray(data=[avg_x, avg_y])
        pub.publish(center_point_msg)

    cv2.imshow('Canny Edge Detection', edge_img)
    cv2.waitKey(1)

if __name__ == '__main__':
    rospy.init_node('lane_detection_node', anonymous=True)
    pub = rospy.Publisher('/center_point', Int32MultiArray, queue_size=1)
    rospy.Subscriber('/usb_cam/image_raw', Image, image_callback)

    rospy.spin()
    cv2.destroyAllWindows()
