import cv2
from matplotlib.pyplot import cla
from yolo import make_yolov3_model
import utils
from weight_reader import WeightReader

INPUT_WIDTH = 416
INPUT_HEIGHT = 416
CLASS_THRESHOLD = 0.5

PHOTO_FILENAME = "images/1.png"


def prepare_yolo():
    yolov3 = make_yolov3_model()

    # load the weights
    weight_reader = WeightReader('weights/yolov3.weights')

    # set the weights
    weight_reader.load_weights(yolov3)

    # save the model to file
    # yolov3.save('model.h5')

    # define the anchors
    anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]

    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    return yolov3, anchors, classes


def yolo_predict(yolov3, anchors, classes, photo_filename, input_width=INPUT_WIDTH, input_height=INPUT_HEIGHT, treshold=CLASS_THRESHOLD):

    image, image_w, image_h = utils.process_image(photo_filename, (input_width, input_height))

    # make prediction
    outs = yolov3.predict(image)

    boxes, class_ids, confidences = list(), list(), list()

    for i in range(len(outs)):
        # decode the output of the network
        current_boxes, current_class_ids, current_confidences = utils.decode_netout(outs[i][0], anchors[i], treshold,
                                                                                    input_height, input_width, image_h,
                                                                                    image_w)
        boxes += current_boxes
        class_ids += current_class_ids
        confidences += current_confidences

    box_indexes = cv2.dnn.NMSBoxes(boxes, confidences, treshold, 0.3)

    return boxes, box_indexes, class_ids, confidences, classes


def yolo_visualize(yolov3, anchors, classes, photo_filename=PHOTO_FILENAME, input_width=INPUT_WIDTH, input_height=INPUT_HEIGHT, treshold=CLASS_THRESHOLD):
    boxes, box_indexes, class_ids, confidences, classes = yolo_predict(yolov3, anchors, classes, photo_filename, input_width, input_height, treshold)
    utils.visualize_boxes(photo_filename, boxes, box_indexes, class_ids, confidences, classes)
