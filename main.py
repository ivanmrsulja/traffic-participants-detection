from numpy.lib.type_check import imag
from evaluation import evaluate_handmade_yolov3, evaluate_preconfigured_yolo, calculate_average_precision, prepare_data_for_handmade, prepare_data_for_pretrained
from yolo import YoloModel
from pretrained_yolo_image import yolo_visualize

if __name__ == '__main__':
    ## Calculate mAP
    # calculate_average_precision(0.2, 0.7)
    # calculate_average_precision(0.2, 0.7, model=YoloModel.V3)
    
    ## Calculate simple metrics for handmade model (accuracy, precision, recall and f-value)
    # yolov3, anchors, classes, image_map = prepare_data_for_handmade()
    # evaluate_handmade_yolov3(yolov3, anchors, classes, image_map, print=True)

    ## Calculate simple metrics for preconfigured model (accuracy, precision, recall and f-value)
    # net, output_layers, colors, classes, image_map = prepare_data_for_pretrained(YoloModel.V4_TINY)
    # evaluate_preconfigured_yolo(net, output_layers, colors, classes, image_map, print=True)

    ## Visualise predictions for image
    yolov3, anchors, classes, image_map = prepare_data_for_handmade()
    yolo_visualize(yolov3, anchors, classes, photo_filename="./images/5.png")
