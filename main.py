from numpy.lib.type_check import imag
from evaluation import evaluate_handmade_yolov3, evaluate_preconfigured_yolo, calculate_average_precision, prepare_data_for_handmade, prepare_data_for_pretrained
from yolo import YoloModel
from pretrained_yolo_image import yolo_visualize
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    
    group.add_argument("-map", "--meanAveragePrecision", help="Calculate mean average precision.\nDefaults to analyzing handmade model.", const="no-value", nargs="?")
    group.add_argument("-smh", "--simpleMetricsHandmade", help="Calculate simple metrics for handmade model (accuracy, precision, recall and f-value)", const="", nargs="?")
    group.add_argument("-smp", "--simpleMetricsPreconfigured", help="Calculate simple metrics for preconfigured model (accuracy, precision, recall and f-value)\nDefaults to 'yolov4-tiny.'", const="yolov4-tiny", nargs="?", type=YoloModel)
    group.add_argument("-vpi", "--visualizePredictionsImage", help="Visualise predictions for image.\nDefaults to 1.png in images folder.", const="./images/1.png", nargs="?")
    
    args = parser.parse_args()

    meanAP = args.meanAveragePrecision
    smh = args.simpleMetricsHandmade
    smp = args.simpleMetricsPreconfigured
    vpi = args.visualizePredictionsImage

    if (not meanAP) and (not smh) and (not smp) and (not vpi):
        print("Run this script with -h or --help flag to see the available options.")
    else: 
        if meanAP:
            if meanAP == "no-value":
                calculate_average_precision(0.2, 0.7)
            else:
                calculate_average_precision(0.2, 0.7, meanAP)

        if smh:
            yolov3, anchors, classes, image_map = prepare_data_for_handmade()
            evaluate_handmade_yolov3(yolov3, anchors, classes, image_map, print=True)
        
        if smp:
            net, output_layers, colors, classes, image_map = prepare_data_for_pretrained(smp)
            evaluate_preconfigured_yolo(net, output_layers, colors, classes, image_map, print=True)

        if vpi:
            yolov3, anchors, classes, image_map = prepare_data_for_handmade()
            yolo_visualize(yolov3, anchors, classes, photo_filename=f"./images/{vpi}")
    
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
    # yolov3, anchors, classes, image_map = prepare_data_for_handmade()
    # yolo_visualize(yolov3, anchors, classes, photo_filename="./images/5.png")
