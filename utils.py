import numpy as np
import cv2
from matplotlib import pyplot
from numpy import expand_dims


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


# TODO: add to util file
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
            # objectness = netout[int(row)][int(col)][b][4] #we dont need this
            # if(objectness.all() <= obj_thresh): continue

            # last elements are class probabilities
            scores = netout[int(row)][col][b][5:]
            class_id = np.argmax(scores)

            if class_id not in [0, 2, 5, 7]: continue  # person, car, bus, truck

            score = scores[class_id]
            if score < 0.5: continue

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
