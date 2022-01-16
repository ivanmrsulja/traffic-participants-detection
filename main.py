from numpy.lib.type_check import imag
from evaluation import evaluate_handmade_yolov3, evaluate_preconfigured_yolo, calculate_average_precision, prepare_data_for_handmade, prepare_data_for_pretrained
from pretrained_yolo_real_time import run_on_real_time_video, load_configured_yolo_model, process_frame_handmade, process_frame_preconfigured
from utils import get_random_colors_for_classes
from yolo import YoloModel
from pretrained_yolo_image import yolo_visualize
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    
    group.add_argument("-map", "--meanAveragePrecision", help="Calculate mean average precision.\nDefaults to analyzing handmade model.", const="no-value", nargs="?", type=YoloModel)
    group.add_argument("-smh", "--simpleMetricsHandmade", help="Calculate simple metrics for handmade model (accuracy, precision, recall and f-value)", const="handmade", nargs="?")
    group.add_argument("-smp", "--simpleMetricsPreconfigured", help="Calculate simple metrics for preconfigured model (accuracy, precision, recall and f-value)\nDefaults to 'yolov4-tiny.'", const="yolov4-tiny", nargs="?", type=YoloModel)
    group.add_argument("-vpi", "--visualizePredictionsImage", help="Visualise predictions for image.\nDefaults to 1.png in images folder.", const="1.png", nargs="?")
    
    videoGroup = group.add_argument_group()
    videoGroup.add_argument("-rrtv", "--runRealTimeVideo",help="Run on video", const="test.mp4", nargs="?")
    videoGroup.add_argument("-a", "--algorithm",help="Choose algorithm to run on video", const="yolov4-tiny", nargs="?", type=YoloModel)

    args = parser.parse_args()

    meanAP = args.meanAveragePrecision
    smh = args.simpleMetricsHandmade
    smp = args.simpleMetricsPreconfigured
    vpi = args.visualizePredictionsImage
    rrtv = args.runRealTimeVideo
    alg = args.algorithm

    if (not meanAP) and (not smh) and (not smp) and (not vpi) and (not rrtv):
        print("Run this script with -h or --help flag to see the available options.")
    else: 
        if meanAP:
            if meanAP == YoloModel.V3_HANDMADE:
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
        
        if rrtv:
            model = None
            if not alg:
                model = YoloModel.V3_HANDMADE
            else:
                model = alg      
            print(f"Using model {model} on video demo_videos/{rrtv}")
            if model == YoloModel.V3_HANDMADE:
                yolov3, anchors, classes, image_map = prepare_data_for_handmade()
                colors = get_random_colors_for_classes(len(classes))
                run_on_real_time_video(yolov3, colors, classes, f"demo_videos/{rrtv}", process_frame_handmade, anchors=anchors)
            else:
                net, output_layers, colors, classes = load_configured_yolo_model(model)
                run_on_real_time_video(net, colors, classes, f"demo_videos/{rrtv}", process_frame_preconfigured, output_layers=output_layers)
    
