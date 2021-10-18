import cv2
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO

from utils.yolo_classes import get_cls_dict
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0


cls_dict = get_cls_dict(80)
vis = BBoxVisualization(cls_dict)
trt_yolo = TrtYOLO('yolov4-tiny-416', 80)
conf_th=0.3

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)
    boxes, confs, clss = trt_yolo.detect(frame, conf_th)
    frame = vis.draw_bboxes(frame, boxes, confs, clss)
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
