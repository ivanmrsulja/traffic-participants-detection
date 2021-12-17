import cv2
import numpy as np
import time
from enum import Enum

class YoloModel(Enum):
    V4_TINY = "yolov4-tiny"
    V3_TINY = "yolov3-tiny"
    V3 = "yolov3"

#TODO: add to yolo file
def load_configured_yolo_model(model):
    net = cv2.dnn.readNet("weights/" + model.value + ".weights", "cfg/" + model.value + ".cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, output_layers, colors, classes

#TODO: add to util file
def decode_net_output(outs, min_confidence, max_iou_for_suppression, width, height):
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            if class_id not in [0, 2, 5, 7]: continue
            confidence = scores[class_id]
            if confidence >= min_confidence:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    box_indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, max_iou_for_suppression)
    return class_ids, confidences, boxes, box_indexes

def run_on_real_time_video(video_path, model, min_confidence=0.5, max_iou_for_suppression=0.3):
    cap = cv2.VideoCapture(video_path)

    net, output_layers, colors, classes = load_configured_yolo_model(model)

    font = cv2.FONT_HERSHEY_PLAIN
    starting_time = time.time()
    frame_count = 0
    while True:
        _, frame = cap.read()
        frame_count += 1

        if frame is None:
            break

        height, width = frame.shape[0], frame.shape[1]

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids, confidences, boxes, box_indexes = decode_net_output(outs, min_confidence, max_iou_for_suppression, width, height)

        for i in range(len(boxes)):
            if i in box_indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 2, color, 2)

        elapsed_time = time.time() - starting_time
        fps = frame_count / elapsed_time
        cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 2, (0, 0, 0), 3)
        cv2.imshow("Demo: {} - {}".format(video_path, model.value), frame)
        key = cv2.waitKey(1)
        if key == 27: # ESCAPE
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_on_real_time_video("demo_videos/test.mp4", YoloModel.V4_TINY)