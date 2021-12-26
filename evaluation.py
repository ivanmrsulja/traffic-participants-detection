from matplotlib.pyplot import cla
from pretrained_yolo_image import prepare_yolo, yolo_predict, yolo_visualize
from utils import bb_intersection_over_union, load_annotated_data
from yolo import load_configured_yolo_model, decode_net_output, load_coco_classes
import cv2


INPUT_WIDTH = 416
INPUT_HEIGHT = 416
CLASS_THRESHOLD = 0.5



def initalize_metrics():
  detection_success = 0
  detection_total = 0

  true_detection_positives = 0
  false_detection_positives = 0
  false_detection_negatives = 0

  true_classification_positives = {
      "car": 0, 
      "bus": 0, 
      "truck": 0, 
      "person": 0
  }
  false_classification_positives = {
      "car": 0, 
      "bus": 0, 
      "truck": 0, 
      "person": 0
  }
  false_classification_negatives = {
      "car": 0, 
      "bus": 0, 
      "truck": 0, 
      "person": 0
  }
  return detection_success, detection_total, true_detection_positives, false_detection_positives, false_detection_negatives, true_classification_positives, false_classification_positives, false_classification_negatives


def print_metrics(detection_success, detection_total, true_detection_positives, false_detection_positives, false_detection_negatives, true_classification_positives, false_classification_positives, false_classification_negatives):
  precision = 0
  if (true_detection_positives > 0):
    precision = true_detection_positives / (true_detection_positives + false_detection_positives)
  recall = 0
  if(true_detection_positives > 0):
    recall = true_detection_positives / (true_detection_positives + false_detection_negatives)
  f_value = 0
  if (precision > 0 or recall > 0):
    f_value = (2 * precision * recall) / (precision + recall)

  print("\n\n==Detection performance measures==\n")
  print(f"\tDetection accuracy: {(detection_success / detection_total) * 100} %")
  print(f"\tDetection precision: {precision * 100} %")
  print(f"\tDetection recall: {recall * 100} %")
  print(f"\tDetection F-Value: {f_value * 100} %")

  print("\n\n==Classification performance measures by class==")
  for item in ['car', 'bus', 'truck', 'person']:
      precision = 0
      if (true_classification_positives[item] > 0):
        precision = true_classification_positives[item] / (true_classification_positives[item] + false_classification_positives[item])
      recall = 0
      if (true_classification_positives[item] > 0):
        recall = true_classification_positives[item] / (true_classification_positives[item] + false_classification_negatives[item])
      f_value = 0
      if (precision > 0 or recall > 0):
        f_value = (2 * precision * recall) / (precision + recall)

      print(f"\n\t=Measures for class {item}=")
      print(f"\t\tPrecision: {precision * 100} %")
      print(f"\t\tRecall: {recall * 100} %")
      print(f"\t\tF-Value: {f_value * 100} %")


def evaluate_handmade_yolov3(iou_threshold=0.5):
    yolov3, anchors, classes = prepare_yolo()
    image_map = load_annotated_data()
    
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
                

            # print(f"Image: {key}:, iou_score: {iou_score} za truth_box: {truth_box}")
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

    print_metrics(detection_success, detection_total, true_detection_positives, false_detection_positives, false_detection_negatives, true_classification_positives, false_classification_positives, false_classification_negatives)


def evaluate_pretrained_yolo(model, iou_threshold=0.5, min_confidence=0.5, max_iou_for_suppression=0.3):
    net, output_layers, colors, classes = load_configured_yolo_model(model)
    image_map = load_annotated_data()
    detection_success, detection_total, true_detection_positives, false_detection_positives, false_detection_negatives, true_classification_positives, false_classification_positives, false_classification_negatives = initalize_metrics()
    classes = load_coco_classes()

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
                

            # print(f"Image: {key}:, iou_score: {iou_score} za truth_box: {truth_box}")
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
    print_metrics(detection_success, detection_total, true_detection_positives, false_detection_positives, false_detection_negatives, true_classification_positives, false_classification_positives, false_classification_negatives)
