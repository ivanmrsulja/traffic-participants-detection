from pretrained_yolo_image import yolo_visualize

INPUT_WIDTH = 416
INPUT_HEIGHT = 416
CLASS_THRESHOLD = 0.5

PHOTO_FILENAME = "images/1.png"


if __name__ == '__main__':
    yolo_visualize(PHOTO_FILENAME, INPUT_WIDTH, INPUT_HEIGHT, CLASS_THRESHOLD)
