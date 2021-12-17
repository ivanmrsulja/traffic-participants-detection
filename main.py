import numpy as np
import struct
import cv2
from numpy import expand_dims
from matplotlib import pyplot
from yolo import make_yolov3_model


#TODO: add to util file
class WeightReader:
	def __init__(self, weight_file):
		with open(weight_file, 'rb') as w_f:
			major,	= struct.unpack('i', w_f.read(4))
			minor,	= struct.unpack('i', w_f.read(4))
			revision, = struct.unpack('i', w_f.read(4))
			if (major*10 + minor) >= 2 and major < 1000 and minor < 1000:
				w_f.read(8)
			else:
				w_f.read(4)
			binary = w_f.read()
		self.offset = 0
		self.all_weights = np.frombuffer(binary, dtype='float32')
 
	def read_bytes(self, size):
		self.offset = self.offset + size
		return self.all_weights[self.offset-size:self.offset]
 
	def load_weights(self, model):
		for i in range(106):
			try:
				conv_layer = model.get_layer('conv_' + str(i))
				print("loading weights of convolution #" + str(i))
				if i not in [81, 93, 105]:
					norm_layer = model.get_layer('bnorm_' + str(i))
					size = np.prod(norm_layer.get_weights()[0].shape)
					beta  = self.read_bytes(size) # bias
					gamma = self.read_bytes(size) # scale
					mean  = self.read_bytes(size) # mean
					var   = self.read_bytes(size) # variance
					norm_layer.set_weights([gamma, beta, mean, var])
				if len(conv_layer.get_weights()) > 1:
					bias   = self.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
					kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
					kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
					kernel = kernel.transpose([2,3,1,0])
					conv_layer.set_weights([kernel, bias])
				else:
					kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
					kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
					kernel = kernel.transpose([2,3,1,0])
					conv_layer.set_weights([kernel])
			except ValueError:
				print("no convolution #" + str(i))

# define the yolo v3 model
yolov3 = make_yolov3_model()

# load the weights
weight_reader = WeightReader('weights/yolov3.weights')

# set the weights
weight_reader.load_weights(yolov3)

# save the model to file
# yolov3.save('model.h5')

#TODO: add to util file
def _sigmoid(x):
  return 1. /(1. + np.exp(-x))

#TODO: add to util file
def decode_netout(netout, anchors, obj_thresh, net_h, net_w, image_h, image_w):
	grid_h, grid_w = netout.shape[:2]
	nb_box = 3
	netout = netout.reshape((grid_h, grid_w, nb_box, -1))

	boxes = []
	class_ids = []
	confidences = []

	netout[Ellipsis, :2]  = _sigmoid(netout[Ellipsis, :2])
	netout[Ellipsis, 4:]  = _sigmoid(netout[Ellipsis, 4:])
	netout[Ellipsis, 5:]  = netout[Ellipsis, 4][Ellipsis, np.newaxis] * netout[Ellipsis, 5:]
	netout[Ellipsis, 5:] *= netout[Ellipsis, 5:] > obj_thresh
 
	for i in range(grid_h*grid_w):
		row = i / grid_w
		col = i % grid_w
		for b in range(nb_box):
			# objectness = netout[int(row)][int(col)][b][4] #we dont need this
			# if(objectness.all() <= obj_thresh): continue
			
			# last elements are class probabilities
			scores = netout[int(row)][col][b][5:]
			class_id = np.argmax(scores)

			if class_id not in [0, 2, 5, 7]: continue # person, car, bus, truck

			score = scores[class_id]
			if score < 0.5: continue

			# first 4 elements are x, y, w, and h
			x, y, w, h = netout[int(row)][int(col)][b][:4]
			x = (col + x) / grid_w 
			y = (row + y) / grid_h 
			w = anchors[2 * b + 0] * np.exp(w) / net_w 
			h = anchors[2 * b + 1] * np.exp(h) / net_h 

			x_real = int((x-w/2) * image_w)
			y_real = int((y-h/2) * image_h)
			w_real = int((x+w/2) * image_w) - x_real
			h_real = int((y+h/2) * image_h) - y_real
			box = [x_real, y_real, w_real, h_real]
			boxes.append(box)
			class_ids.append(class_id)
			confidences.append(float(score))
	return boxes, class_ids, confidences


# define the anchors
anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]

# define the probability threshold for detected objects
class_threshold = 0.5

# load and prepare an image
def load_image(filename, shape):
    image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    width, height = image.shape[1],image.shape[0] 
    image = cv2.resize(image, shape, interpolation = cv2.INTER_AREA)
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height

photo_filename = "images/1.png"

with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

# define the expected input shape for the model
input_w, input_h = 416, 416

image, image_w, image_h = load_image(photo_filename, (input_w, input_h))

# make prediction
outs = yolov3.predict(image)

boxes, class_ids, confidences = list(), list(), list() 
for i in range(len(outs)):
# decode the output of the network
	current_boxes, current_class_ids, current_confidences = decode_netout(outs[i][0], anchors[i], class_threshold, input_h, input_w, image_h, image_w)
	boxes += current_boxes
	class_ids += current_class_ids
	confidences += current_confidences

box_indexes = cv2.dnn.NMSBoxes(boxes, confidences, class_threshold, 0.3)

def visualize_boxes(filename, boxes, box_ids, label_indexes, scores, label_names):
	image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
	for idx in box_ids:
		x, y, w, h = boxes[idx[0]]
		cv2.rectangle(image, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)
		label = "{0} ({1:.5g})".format(label_names[label_indexes[idx[0]]], scores[idx[0]])
		cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
	pyplot.imshow(image)
	pyplot.show()

visualize_boxes(photo_filename, boxes, box_indexes, class_ids, confidences, classes)