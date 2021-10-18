"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""

from icecream import ic
import os
import time
import uuid
from numpy.lib.npyio import save
import pika
import base64
import numpy as np
from imageio import imread
import io
import json
import ast
from threading import Thread
import queue
import datetime
q=queue.Queue()

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver
import Jetson.GPIO as GPIO
import RPi.GPIO as GPIO
from obj_tracking import CentroidTracker

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO
from micro_services.request import addNewInference

def __init__(self, cls_dict):
    self.cls_dict = cls_dict
WINDOW_NAME = 'TrtYOLODemo'


# def rabbitMQ(data):
#     connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
#     channel = connection.channel()
#     channel.queue_declare(queue='in')

#     # Send the image tho rabbitMQ
#     channel.basic_publish(exchange='',  
#             routing_key='in',
#             body=json.dumps(data))

def show_image(cam):

    ret, frame = cam.read()

    cv2.imshow('frame', frame)

    cv2.waitKey(25)


def loop_and_detect(cam, trt_yolo, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    full_scrn = False
    fps = 0.0
    tic = time.time()
    

    ct = CentroidTracker()

    auxId = -1
    save_frames = []

    n = 0

    port = 33
    GPIO.cleanup()
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(port, GPIO.IN)
    controll = True

    while True:
        ret, frame = cam.read()

        if frame is None or ret is not True:
                continue
        
        try:  

            save_frames.append(frame)

            result = GPIO.input(port)
            aux = True

            # cv2.imshow('frame', frame)

            show_image(cam) 

            while not result:
                show_image(cam)            
                result = GPIO.input(port)

                if controll:
                    boxes, confs, clss = trt_yolo.detect(save_frames[0], conf_th)
                    results = vis.draw_bboxes(save_frames[0], boxes, confs, clss)
                    frame = results[0]
                    # frame = show_fps(frame, fps)

                    cls_image = results[1]
                    newBoxes = results[2]
                    print(newBoxes)
                    print(cls_image)
                    
                    # Convert captured image to JPG
                    ret, buffer = cv2.imencode('.jpg', frame)

                    # Convert to base64 encoding and show start of data
                    base = base64.b64encode(buffer).decode('utf-8')
                    data = {
                        0: frame,
                        1: cls_image,
                        2: confs.tolist()
                    }
            
                    Thread(target=addNewInference, args=(data,)).start()
                    controll = False

                
                save_frames = []          
            
            controll = True


            k = cv2.waitKey(1)
            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break

        except Exception as e:
            print(str(e), 'loop error')
            
            continue



def main():
    
    # yolo configs
    category_num = 80
    model = 'yolov4-tiny-416'

    cls_dict = get_cls_dict(category_num)
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(model, category_num)

    # Connect to camera
    cam = cv2.VideoCapture() 
    cam.open("rtsp://admin:Pti2389!!@192.168.0.100:554/h264/ch1/sub")
    # cam.open("rtsp://admin:Pti2389!!@192.168.1.64:554/h264/ch1/sub")


    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')
    
    # Thread(target = loop_and_detect(cam, trt_yolo, 0.3, vis)).start()

    # Loop to detect
    loop_and_detect(cam, trt_yolo, conf_th=0.3, vis=vis)

    # Close all
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
