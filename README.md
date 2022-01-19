# traffic-participants-detection

Traffic participants detection, localization and classification on real time video.

## Installation

All required packages can be found in requirements.txt file. You can install them using pip:

`pip install -r requirements.txt`.

It is recommended that you do that inside a [virtual environment](https://docs.python.org/3/tutorial/venv.html).

## Data

Files necessary for testing and running are inside zip archives which you can download from Google Drive. After you download these zip archives, extract them on the project's root level.

Archives:

- [demo_videos](https://drive.google.com/file/d/1IkQJcNYo4uXjHGW2AVfFj4x9M9H9s3-V/view?usp=sharing)
- [images](https://drive.google.com/file/d/1U7lpKty4WrqENrBfXwfcCTJsgW3yk8gp/view?usp=sharing)
- [weights](https://drive.google.com/file/d/1aR09gku6OqK669yO9B1FNuDmHiniDqsP/view?usp=sharing)

## Running the project

You can run the project like any other python script:

`python main.py args...`

If you are using Linux, you may need to specify python version as well.

args... is only a placeholder for command line arguments.

### Command line arguments

Only -rrtv and -a arguments can be placed alongside each other.

These options can also be viewed by running the `main.py` file with `-h` flag enabled.

| Short option | Long option                  | Description                                                                                                             |
| ------------ | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| -map         | --meanAveragePrecision       | Calculates mean average precision.Defaults to analyzing handmade model.                                                 |
| -smh         | --simpleMetricsHandmade      | Calculates simple metrics for handmade model (accuracy, precision, recall and f-value).                                 |
| -smp         | --simpleMetricsPreconfigured | Calculates simple metrics for preconfigured model (accuracy, precision, recall and f-value). Defaults to 'yolov4-tiny.' |
| -vpi         | --visualizePredictionsImage  | Visualize predictions for image. Defaults to '1.png' in images folder.                                                  |
| -rrtv        | --runRealTimeVideo           | Run on video. Defaults to 'test.mp4' in demo_videos folder.                                                             |
| -a           | --algorithm                  | Choose algorithm to run on video. Defaults to 'yolov4-tiny'.                                                            |

## Demos

TODO: add demos

## Sources

- https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173

- https://github.com/muhammadshiraz/YOLO-Real-Time-Object-Detection

- https://github.com/experiencor/keras-yolo3

- https://blog.roboflow.com/a-thorough-breakdown-of-yolov4/

- https://towardsdatascience.com/yolo-v3-explained-ff5b850390f

- https://towardsdatascience.com/yolo-v3-object-detection-with-keras-461d2cfccef6

- https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/

- https://towardsdatascience.com/object-detection-using-yolov3-and-opencv-19ee0792a420

- https://www.youtube.com/watch?v=9s_FpMpdYW8

- https://www.youtube.com/watch?v=RTlwl2bv0Tg
