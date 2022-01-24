import cv2
import numpy as np
import time
from yolo import YoloModel, load_configured_yolo_model, decode_net_output
from evaluation import prepare_data_for_handmade
from utils import decode_netout
from numpy import expand_dims


def run_on_real_time_video(model, colors, classes, video_path, process_function, output_layers=None, anchors=None, min_confidence=0.5, max_iou_for_suppression=0.3):
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

        boxes, box_indexes, class_ids, confidences = process_function(model, frame, min_confidence, height, width, 
                                                                    max_iou_for_suppression, output_layers=output_layers, anchors=anchors)

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
        if elapsed_time >= 1:
            starting_time = time.time()
            frame_count = 0
        cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 2, (0, 0, 0), 3)
        cv2.imshow("Demo: {}".format(video_path), frame)
        key = cv2.waitKey(1)
        if key == 27: # ESCAPE
            break
    cap.release()
    cv2.destroyAllWindows()


def process_frame_handmade(model, frame, min_confidence, height, width, max_iou_for_suppression, output_layers=None, anchors=None):
    image = cv2.resize(frame, (416, 416), interpolation=cv2.INTER_AREA)
    image = image.astype('float32')
    image /= 255.0
    image = expand_dims(image, 0)

    outs = model.predict(image)

    boxes, class_ids, confidences = list(), list(), list()

    for i in range(len(outs)):
        # decode the output of the network
        current_boxes, current_class_ids, current_confidences = decode_netout(outs[i][0], anchors[i], min_confidence,
                                                                                    416, 416, height,
                                                                                    width)
        boxes += current_boxes
        class_ids += current_class_ids
        confidences += current_confidences

    box_indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, max_iou_for_suppression)

    return boxes, box_indexes, class_ids, confidences


def process_frame_preconfigured(model, frame, min_confidence, height, width, max_iou_for_suppression, output_layers=None, anchors=None):
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    model.setInput(blob)
    outs = model.forward(output_layers)

    return decode_net_output(outs, min_confidence, max_iou_for_suppression, width, height)


if __name__ == "__main__":
    net, output_layers, colors, classes = load_configured_yolo_model(YoloModel.V4_TINY)
    run_on_real_time_video(net, colors, classes, "demo_videos/test.mp4", process_frame_preconfigured, output_layers=output_layers)

    # yolov3, anchors, classes, image_map = prepare_data_for_handmade()
    # colors = np.random.uniform(0, 255, size=(len(classes), 3))
    # run_on_real_time_video(yolov3, colors, classes, "demo_videos/test.mp4", process_frame_handmade, anchors=anchors)
    