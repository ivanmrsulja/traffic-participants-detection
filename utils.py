import numpy as np
import cv2
from matplotlib import pyplot
from numpy import expand_dims


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def get_random_colors_for_classes(num_classes):
    return np.random.uniform(0, 255, size=(num_classes, 3))


def decode_netout(netout, anchors, obj_thresh, net_h, net_w, image_h, image_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))

    boxes = []
    class_ids = []
    confidences = []

    netout[Ellipsis, :2] = sigmoid(netout[Ellipsis, :2])
    netout[Ellipsis, 4:] = sigmoid(netout[Ellipsis, 4:])
    netout[Ellipsis, 5:] = netout[Ellipsis, 4][Ellipsis, np.newaxis] * netout[Ellipsis, 5:]
    netout[Ellipsis, 5:] *= netout[Ellipsis, 5:] > obj_thresh

    for i in range(grid_h * grid_w):
        row = i / grid_w
        col = i % grid_w
        for b in range(nb_box):
            objectness = netout[int(row)][int(col)][b][4] #we dont need this
            if(objectness.all() <= obj_thresh): continue

            # last elements are class probabilities
            scores = netout[int(row)][col][b][5:]
            class_id = np.argmax(scores)

            if class_id not in [0, 2, 5, 7]: continue  # person, car, bus, truck

            score = scores[class_id]
            # if score < obj_thresh: continue

            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]
            x = (col + x) / grid_w
            y = (row + y) / grid_h
            w = anchors[2 * b + 0] * np.exp(w) / net_w
            h = anchors[2 * b + 1] * np.exp(h) / net_h

            x_real = int((x - w / 2) * image_w)
            y_real = int((y - h / 2) * image_h)
            w_real = int((x + w / 2) * image_w) - x_real
            h_real = int((y + h / 2) * image_h) - y_real
            box = [x_real, y_real, w_real, h_real]
            boxes.append(box)
            class_ids.append(class_id)
            confidences.append(float(score))
    return boxes, class_ids, confidences


# load and prepare an image
def process_image(filename, shape):
    image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    width, height = image.shape[1], image.shape[0]
    image = cv2.resize(image, shape, interpolation=cv2.INTER_AREA)
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height


def visualize_boxes(filename, boxes, box_ids, label_indexes, scores, label_names):
    image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    for idx in box_ids:
        x, y, w, h = boxes[idx[0]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)
        label = "{0} ({1:.5g})".format(label_names[label_indexes[idx[0]]], scores[idx[0]])
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    pyplot.imshow(image)
    pyplot.show()


def load_annotated_data():
    image_map = {}
    with open('./images-annotation-data/labels_test.csv', 'r') as file:
        lines = file.readlines()
        for line in lines:
            tokens = line.split(',')
            image_class, x, y, w, h, image_name, image_height, image_width = tokens
            if image_name not in image_map.keys():
                image_map[image_name] = []
            image_map[image_name].append((image_class, int(x), int(y), int(w), int(h)))
    return image_map


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


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


def print_metrics(result_map):
    print("\n\n==Recognition performance measures==\n")
    print(f"\tRecognition accuracy: {result_map['detection'][0] * 100} %")
    print(f"\tRecognition precision: {result_map['detection'][1] * 100} %")
    print(f"\tRecognition recall: {result_map['detection'][2] * 100} %")
    print(f"\tRecognition F-Value: {result_map['detection'][3] * 100} %")

    print("\n\n==Classification performance measures by class==")
    for key in result_map["classification"]:
        print(f"\n\t=Measures for class {key}=")
        print(f"\t\tPrecision: {result_map['classification'][key][0] * 100} %")
        print(f"\t\tRecall: {result_map['classification'][key][1] * 100} %")
        print(f"\t\tF-Value: {result_map['classification'][key][2] * 100} %")


def calculate_metrics(detection_success, detection_total,
    true_detection_positives, false_detection_positives, 
    false_detection_negatives, true_classification_positives, 
    false_classification_positives, false_classification_negatives):

    detection_precision = 0
    if (true_detection_positives > 0):
        detection_precision = true_detection_positives / (true_detection_positives + false_detection_positives)
    detection_recall = 0
    if(true_detection_positives > 0):
        detection_recall = true_detection_positives / (true_detection_positives + false_detection_negatives)
    detection_f_value = 0
    if (detection_precision > 0 or detection_recall > 0):
        detection_f_value = (2 * detection_precision * detection_recall) / (detection_precision + detection_recall)
    
    detection_accuracy = detection_success / detection_total

    result_map = {
        "detection": [detection_accuracy, detection_precision, detection_recall, detection_f_value], 
        "classification": {
            "car": [], 
            "bus": [], 
            "truck": [], 
            "person": []
        }
    }

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
        
        result_map["classification"][item] = [precision, recall, f_value]

    return result_map
    