import cv2
import numpy as np
import time
from yolo import YoloModel, load_configured_yolo_model


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
    # return class_ids, confidences, boxes, box_indexes
    return boxes, box_indexes, class_ids, confidences

def load_yolo(model):
    return load_configured_yolo_model(model)



def run_on_real_time_video(net, output_layers, colors, classes, video_path, min_confidence=0.5, max_iou_for_suppression=0.3):
    cap = cv2.VideoCapture(video_path)

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

        boxes, box_indexes, class_ids, confidences = decode_net_output(outs, min_confidence, max_iou_for_suppression, width, height)

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
        cv2.imshow("Demo: {}".format(video_path), frame)
        key = cv2.waitKey(1)
        if key == 27: # ESCAPE
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    net, output_layers, colors, classes = load_yolo(YoloModel.V4_TINY)
    run_on_real_time_video(net, output_layers, colors, classes,"demo_videos/test.mp4")