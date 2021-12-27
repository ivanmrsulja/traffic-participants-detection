from matplotlib.pyplot import cla
from pretrained_yolo_image import prepare_yolo, yolo_predict, yolo_visualize
from utils import bb_intersection_over_union, load_annotated_data, initalize_metrics, calculate_metrics, print_metrics
from yolo import load_configured_yolo_model, decode_net_output, load_coco_classes
import cv2
import numpy as np

INPUT_WIDTH = 416
INPUT_HEIGHT = 416
CLASS_THRESHOLD = 0.5


def prepare_data_for_handmade():
    yolov3, anchors, classes = prepare_yolo()
    image_map = load_annotated_data()
    return yolov3, anchors, classes, image_map


def prepare_data_for_pretrained(model):
    net, output_layers, colors, classes = load_configured_yolo_model(model)
    image_map = load_annotated_data()
    return net, output_layers, colors, classes, image_map


def evaluate_handmade_yolov3(yolov3, anchors, classes, image_map, iou_threshold=0.5, print=False):
   
    detection_success, detection_total, true_detection_positives, false_detection_positives, false_detection_negatives, true_classification_positives, false_classification_positives, false_classification_negatives = initalize_metrics()
    
    for key in image_map.keys():
        boxes, box_indexes, class_ids, confidences, classes = yolo_predict(yolov3, anchors, classes, f"./images/{key}")
        correct_boxes = [boxes[i] for i in range(len(boxes)) if i in box_indexes]
        correct_confidences = [confidences[i] for i in range(len(confidences)) if i in box_indexes]
        correct_class_ids = [class_ids[i] for i in range(len(class_ids)) if i in box_indexes]  
        for truth_box in image_map[key]:
            iou_score = 0
            box_index = 0
            truth_box_tranformed = (truth_box[1], truth_box[2], truth_box[1] + truth_box[3], truth_box[2] + truth_box[4])
            detection_total += 1
            for i in range(len(correct_boxes)):
                box = correct_boxes[i]
                predicted_box_transformed = (box[0], box[1], box[0] + box[2], box[1] + box[3])
                calculated_iou_score = bb_intersection_over_union(truth_box_tranformed, predicted_box_transformed)
                if calculated_iou_score > iou_score:
                    iou_score = calculated_iou_score
                    box_index = i
                
            if iou_score >= iou_threshold:
                detection_success += 1
                true_detection_positives += 1
                if truth_box[0] == classes[correct_class_ids[box_index]]:
                    true_classification_positives[truth_box[0]] += 1
                else:
                    false_classification_positives[truth_box[0]] += 1

                del correct_boxes[box_index]
                del correct_confidences[box_index]
                del correct_class_ids[box_index]
            else:
                false_detection_negatives += 1
                false_classification_negatives[truth_box[0]] += 1

        false_detection_positives += len(correct_boxes)

    result_map = calculate_metrics(detection_success, detection_total, true_detection_positives, false_detection_positives, false_detection_negatives, true_classification_positives, false_classification_positives, false_classification_negatives)
    if(print):
        print_metrics(result_map)
    return result_map


def evaluate_preconfigured_yolo(net, output_layers, _, classes, image_map, iou_threshold=0.5, min_confidence=0.5, max_iou_for_suppression=0.3, print=False):
    detection_success, detection_total, true_detection_positives, false_detection_positives, false_detection_negatives, true_classification_positives, false_classification_positives, false_classification_negatives = initalize_metrics()

    for key in image_map.keys():
        image = cv2.cvtColor(cv2.imread(f"./images/{key}"), cv2.COLOR_BGR2RGB)
        height, width = image.shape[0], image.shape[1]
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        boxes, box_indexes, class_ids, confidences = decode_net_output(outs, min_confidence, max_iou_for_suppression, width, height)
        correct_boxes = [boxes[i] for i in range(len(boxes)) if i in box_indexes]
        correct_confidences = [confidences[i] for i in range(len(confidences)) if i in box_indexes]
        correct_class_ids = [class_ids[i] for i in range(len(class_ids)) if i in box_indexes]  
        for truth_box in image_map[key]:
            iou_score = 0
            box_index = 0
            truth_box_tranformed = (truth_box[1], truth_box[2], truth_box[1] + truth_box[3], truth_box[2] + truth_box[4])
            detection_total += 1
            for i in range(len(correct_boxes)):
                box = correct_boxes[i]
                predicted_box_transformed = (box[0], box[1], box[0] + box[2], box[1] + box[3])
                calculated_iou_score = bb_intersection_over_union(truth_box_tranformed, predicted_box_transformed)
                if calculated_iou_score > iou_score:
                    iou_score = calculated_iou_score
                    box_index = i

            if iou_score >= iou_threshold:
                detection_success += 1
                true_detection_positives += 1
                if truth_box[0] == classes[correct_class_ids[box_index]]:
                    true_classification_positives[truth_box[0]] += 1
                else:
                    false_classification_positives[truth_box[0]] += 1

                del correct_boxes[box_index]
                del correct_confidences[box_index]
                del correct_class_ids[box_index]
            else:
                false_detection_negatives += 1
                false_classification_negatives[truth_box[0]] += 1

        false_detection_positives += len(correct_boxes)
    result_map = calculate_metrics(detection_success, detection_total, true_detection_positives, false_detection_positives, false_detection_negatives, true_classification_positives, false_classification_positives, false_classification_negatives)
    if(print):
        print_metrics(result_map)
    return result_map


def calculate_average_precision(start, finish, model=None):
    thresholds = np.arange(start=start, stop=finish, step=0.05)
    evaluation_map = {
        "car": {
            "precision": [],
            "recall": []
        }, 
        "bus": {
            "precision": [],
            "recall": []
        }, 
        "truck": {
            "precision": [],
            "recall": []
        }, 
        "person": {
            "precision": [],
            "recall": []
        }
    }
    
    if model is None:
        yolov3, anchors, classes, image_map = prepare_data_for_handmade()
    else:
        net, output_layers, colors, classes, image_map = prepare_data_for_pretrained(model)

    for count, thresh in enumerate(thresholds):
        result_map = {}
        if model is None:
            result_map = evaluate_handmade_yolov3(yolov3, anchors, classes, image_map, thresh)
        else:
            result_map = evaluate_preconfigured_yolo(net, output_layers, colors, classes, image_map, thresh)
        for key in result_map["classification"].keys():
            item = result_map["classification"][key]
            evaluation_map[key]["precision"].append(item[0])
            evaluation_map[key]["recall"].append(item[1])
        print(f"==Finished iteration {count + 1} of {len(thresholds)}")

    print("==Average precisions")
    total_ap = 0
    for key in evaluation_map:
        evaluation_map[key]["precision"].append(1.0)
        evaluation_map[key]["recall"].append(0.0)

        ap = 0
        for count, thresh in enumerate(thresholds):
            ap += (evaluation_map[key]["recall"][count] - evaluation_map[key]["recall"][count + 1]) * evaluation_map[key]["precision"][count]
        
        print(f"\t=Average precision for class {key}: {ap}")
        total_ap += ap
    
    print(f"==Mean average precision: {total_ap / 4}")