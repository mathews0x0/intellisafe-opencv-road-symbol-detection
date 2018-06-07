__author__ = 'zhengwang'

import threading
import SocketServer
import serial
import cv2
import numpy as np
import math


class DistanceToCamera(object):

    def __init__(self):
        # camera params
        self.alpha = 8.0 * math.pi / 180
        self.v0 = 119.865631204
        self.ay = 332.262498472

    def calculate(self, v, h, x_shift, image):
        # compute and return the distance from the target point to the camera
        d = h / math.tan(self.alpha + math.atan((v - self.v0) / self.ay))
        if d > 0:
            cv2.putText(image, "%.1fcm" % d,
                (image.shape[1] - x_shift, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return d


class ObjectDetection(object):

    def __init__(self):
        self.red_light = False
        self.green_light = False
        self.yellow_light = False

    def detect(self, cascade_classifier, gray_image, image):

        # y camera coordinate of the target point 'P'
        v = 0

        # detection
        cascade_obj = cascade_classifier.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        # draw a rectangle around the objects
        for (x_pos, y_pos, width, height) in cascade_obj:
            cv2.rectangle(image, (x_pos+5, y_pos+5), (x_pos+width-5, y_pos+height-5), (255, 255, 255), 2)
            v = y_pos + height - 5
           
            roi = gray_image[y_pos+10:y_pos + height-10, x_pos+10:x_pos + width-10]
            mask = cv2.GaussianBlur(roi, (25, 25), 0)
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(mask)
                

    
        return v




class VideoStreamHandler(SocketServer.StreamRequestHandler):

    # h1: stop sign
    h1 = 15.5 - 10  # cm

    # create neural network
    model = NeuralNetwork()
    model.create()

    obj_detection = ObjectDetection()
    rc_car = RCControl()

    # cascade classifiers
    stop_cascade = cv2.CascadeClassifier('cascade_xml/stop_sign.xml')
    light_cascade = cv2.CascadeClassifier('cascade_xml/traffic_light.xml')
    limit_cascade =  cv2.CascadeClassifier('cascade_xml/speed_limit.xml')

    d_to_camera = DistanceToCamera()
    d_stop_sign = 25
    d_light = 25
    d_limit = 25

   
    def handle(self):

        global sensor_data
        stream_bytes = ' '
        stop_flag = False
        stop_sign_active = False


        # stream video frames one by one
        try:
            while True:
                stream_bytes += self.rfile.read(1024)
                first = stream_bytes.find('\xff\xd8')
                last = stream_bytes.find('\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last+2]
                    stream_bytes = stream_bytes[last+2:]
                    gray = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.CV_LOAD_IMAGE_GRAYSCALE)
                    image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)

                    # object detection
                    v_param1 = self.obj_detection.detect(self.stop_cascade, gray, image)
                    v_param2 = self.obj_detection.detect(self.light_cascade, gray, image)
                    v_param3 = self.obj_detection.detect(self.limit_cascade, gray, image)


                    # distance measurement
                    if v_param1 > 0 or v_param2 > 0 or v_param3 > 0:
                        d1 = self.d_to_camera.calculate(v_param1, self.h1, 300, image)
                        d2 = self.d_to_camera.calculate(v_param2, self.h2, 100, image)
                        d3 = self.d_to_camera.calculate(v_param3, self.h2, 100, image)
                        self.d_stop_sign = d1
                        self.d_light = d2
                        self.d_limit = d3

                    cv2.imshow('image', image)
                   

                    # reshape image
                    image_array = half_gray.reshape(1, 38400).astype(np.float32)
                    
                 
                    # stop conditions
                    
                    if  0 < self.d_stop_sign <25 and  :
                        print("Stop sign ahead")
                    elif 0< self.d_limit < 25 :
                        print("speed limit detected")

                                       
            cv2.destroyAllWindows()

        finally:
            print "Connection closed on thread 1"


class ThreadServer(object):

    def server_thread(host, port):
        server = SocketServer.TCPServer((host, port), VideoStreamHandler)
        server.serve_forever()

    video_thread = threading.Thread(target=server_thread('192.168.43.41', 8050))
    video_thread.start()

if __name__ == '__main__':
    ThreadServer()
