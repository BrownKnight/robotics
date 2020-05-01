# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Human Following
# This notebook uses the COCO pretrained model to detect the location of people in front of it, then move towards the cloest person - as determined by the robots depth camera

# ### Load the pre-trained object detection model
# When the model is used later on, we will only look at the "person" category even though the model will output bounding boxes for all sorts of objects.

# + pycharm={"is_executing": false}
import tensorrt as trt
from tensorrt_model import TRTModel
from ssd_tensorrt import load_plugins, parse_boxes, TRT_INPUT_NAME, TRT_OUTPUT_NAME
import numpy as np

from Camera import Camera

mean = 255.0 * np.array([0.5, 0.5, 0.5])
stdev = 255.0 * np.array([0.5, 0.5, 0.5])


def bgr8_to_ssd_input(camera_value):
    x = camera_value
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1)).astype(np.float32)
    x -= mean[:, None, None]
    x /= stdev[:, None, None]
    return x[None, ...]


class ObjectDetector(object):

    def __init__(self, engine_path, preprocess_fn=bgr8_to_ssd_input):
        logger = trt.Logger()
        trt.init_libnvinfer_plugins(logger, '')
        load_plugins()
        self.trt_model = TRTModel(engine_path, input_names=[TRT_INPUT_NAME],
                                  output_names=[TRT_OUTPUT_NAME, TRT_OUTPUT_NAME + '_1'])
        self.preprocess_fn = preprocess_fn

    def execute(self, *inputs):
        trt_outputs = self.trt_model(self.preprocess_fn(*inputs))
        return parse_boxes(trt_outputs)

    def __call__(self, *inputs):
        return self.execute(*inputs)


model = ObjectDetector('ssd_mobilenet_v2_coco.engine')

# -

# ### Initialize the camera instance for the Intel realsense sensor D435i

# + pycharm={"is_executing": false}
# use traitlets and widgets to display the image in Jupyter Notebook

# use opencv to covert the depth image to RGB image for displaying purpose
import cv2
import numpy as np

# using realsense to capture the color and depth image

# multi-threading is used to capture the image in real time performance


def bgr8_to_jpeg(value, quality=75):
    return bytes(cv2.imencode('.jpg', value)[1])


# create a camera object
# camera = Camera.instance()
# camera.start()  # start capturing the data
camera = Camera.instance()
camera.start()
# -

# ### Run the model on the camera inpout, and move towards people
# Human is labeled is 1 in the pretrained model. A full list of the detection class indices can be found in the following 
# link https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_complete_label_map.pbtxt
#
# The program will look at all the people found in the image, and determine using the cameras depth sensor where the 
# location of the closest person is. It will then move towards the person using the centre x coordinate to determine how 
# far left/right to move. It alos uses the distance to determine how fast it should move towards/away from the person. 
# It will stop within a given distance in front of the person.

# + pycharm={"is_executing": false}
import ipywidgets.widgets as widgets
from IPython.display import display

width = 640
height = 480

image_widget = widgets.Image(format='jpeg')
display(image_widget)

import time
from RobotClass import Robot

# initialize the Robot class
robot = Robot()


def normalize_distance(value):
    bounds = {'actual': {'lower': 0, 'upper': 4000}, 'desired': {'lower': 60, 'upper': 460}}
    return int(bounds['desired']['lower'] + (value - bounds['actual']['lower']) * (
                bounds['desired']['upper'] - bounds['desired']['lower']) / (
                           bounds['actual']['upper'] - bounds['actual']['lower']))


def processing(change):
    image = change['new']

    imgsized = cv2.resize(image, (300, 300))
    # compute all detected objects
    detections = model(imgsized)
    # DEBUG
    # detections = [[{'label':None}]]

    matching_detections = [d for d in detections[0] if d['label'] == 1]

    # Draw some lines on the image to show the boundary lines
    fast_left = 220
    slow_left = 270
    slow_right = 380
    fast_right = 420

    # Values are the bottom of the range i.e. will move fast backwards if less than distance_really_close
    distance_really_close = 1800
    distance_close = 2200
    distance_stable = 2600
    distance_far = 3000
    distance_really_far = 3600

    draw_guidance_lines(fast_left, slow_left, fast_right, slow_right, distance_really_close, distance_close,
                        distance_stable, distance_far, distance_really_far, image)

    closest_person = (0, 9999, 9999)
    for index, det in enumerate(matching_detections):
        bbox = det['bbox']
        x = int(width * bbox[0])
        y = int(height * bbox[1])
        x1 = int(width * bbox[2])
        y1 = int(height * bbox[3])
        half_height = int((y1 - y) / 2)
        quarter_width = int((x1 - x) / 4)
        centre = (x + x1) / 2
        # Draw a blue rectangle around every person it detects
        cv2.rectangle(image, (x, y), (x1, y1), (255, 0, 0), 2)
        # Draw a thin blue rectangle around the area of the bounding box we actually use to determine the distance
        cv2.rectangle(image, (x + quarter_width, y), (x1 - quarter_width, y1 - half_height), (150, 10, 10), 2)
        distance = np.average(camera.depth_image[y:y1 - half_height, x + quarter_width:x1 - quarter_width])
        cv2.putText(image, "D %.0f" % distance, (x + 10, y + half_height), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255),
                    2)
        closest_person = ((x, y, x1, y1), centre, distance) if distance < closest_person[2] else closest_person

    speed = 0
    turn_speed = 0.2
    # Determine how to move
    if len(matching_detections) != 0:
        x, y, x1, y1 = closest_person[0]
        distance = closest_person[2]
        cv2.circle(image, (40, normalize_distance(distance)), 4, (0, 165, 255), -1)
        cv2.rectangle(image, (x, y), (x1, y1), (255, 255, 255), 3)
        centre = closest_person[1]
        cv2.circle(image, (int(centre), 40), 4, (0, 165, 255), -1)

        if centre < fast_left:
            turn_speed = -0.7
        elif centre < slow_left:
            turn_speed = -0.4
        elif slow_left < centre < slow_right:
            turn_speed = 0.0
        elif slow_right < centre < fast_right:
            turn_speed = 0.4
        elif centre >= fast_right:
            turn_speed = 0.7

        if distance < distance_really_close:
            speed = -0.5
        elif distance < distance_close:
            speed = -0.2
        elif distance < distance_stable:
            speed = 0
        elif distance < distance_far:
            speed = 0.2
        elif distance < distance_really_far:
            speed = 0.5
        else:
            speed = 0.6

    left_speed = speed + turn_speed
    right_speed = speed - turn_speed
    robot.set_motors(left_speed, right_speed, left_speed, right_speed)

    image_widget.value = bgr8_to_jpeg(image)


def draw_guidance_lines(fast_left, slow_left, fast_right, slow_right, distance_really_close, distance_close,
                        distance_stable, distance_far, distance_really_far, image):
    # Draw guidance lines at top of the screen for moving left/right
    colour = (0, 165, 255)
    add_direction_guideline(colour, fast_left, image, "<<<<")
    add_direction_guideline(colour, slow_left, image, "<<")
    add_direction_guideline(colour, slow_right, image, "----")
    add_direction_guideline(colour, fast_right, image, ">>")
    cv2.putText(image, ">>>>", (fast_right + 40, 20), cv2.FONT_HERSHEY_PLAIN, 1, colour, 1)


    # Draw guidance on left side of screen for distance markers
    top_boundary = 60
    bottom_boundary = 460
    cv2.line(image, (5, top_boundary), (55, top_boundary), colour, 2)
    add_distance_guideline(colour, distance_really_close, image, "----")
    add_distance_guideline(colour, distance_close, image, "--")
    add_distance_guideline(colour, distance_stable, image, "STOP")
    add_distance_guideline(colour, distance_far, image, "++")
    add_distance_guideline(colour, distance_really_far, image, "++++")
    cv2.line(image, (5, bottom_boundary), (55, bottom_boundary), colour, 2)


def add_direction_guideline(colour, x_coord, image, string):
    cv2.line(image, (x_coord, 20), (x_coord, 60), colour, 1)
    cv2.putText(image, string, (x_coord - 16*len(string), 20), cv2.FONT_HERSHEY_PLAIN, 1, colour, 1)


def add_distance_guideline(colour, distance, image, string):
    normalised = normalize_distance(distance)
    cv2.line(image, (20, normalised), (40, normalised), colour, 1)
    cv2.putText(image, string, (45, normalised-15), cv2.FONT_HERSHEY_PLAIN, 1, colour, 1)


# the camera.observe function will monitor the color_value variable. If this value changes, the excecute function will be excuted.
processing({'new': camera.color_value})
camera.observe(processing, names='color_value')
# -

# The following code can be used to stop the capturing of the image and the moving of the robot

# + pycharm={"is_executing": false}
camera.unobserve_all()
time.sleep(1.0)
robot.stop()
# -

# ## Tasks
#
# ### 1. Please try to calculate the distance between the detected human and the robot using the depth image. (Note: You can refer to tutorial 2 to get the depth information for a specific point)
# ### 2. Please try to add collision avoidance function into this program to protect the robot. (Note: You can refer to tutorial 4 for the collision avoidance)
# ### 3. Think about how to control the robot to move towards the detected human
