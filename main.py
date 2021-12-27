from numpy.lib.type_check import imag
from evaluation import evaluate_handmade_yolov3, evaluate_pretrained_yolo, calculate_average_precision
from yolo import YoloModel

if __name__ == '__main__':
    calculate_average_precision(0.2, 0.7, model=YoloModel.V3_TINY)
    #evaluate_handmade_yolov3(print=True)
    #evaluate_pretrained_yolo(YoloModel.V4_TINY, print=True)
    