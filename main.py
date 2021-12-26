from numpy.lib.type_check import imag
from evaluation import evaluate_handmade_yolov3, evaluate_pretrained_yolo
from yolo import YoloModel

if __name__ == '__main__':
    evaluate_handmade_yolov3()
    #evaluate_pretrained_yolo(YoloModel.V4_TINY)
    