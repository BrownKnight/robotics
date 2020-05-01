import threading

import cv2
import numpy as np
import traitlets
from traitlets.config import SingletonConfigurable


class Camera(SingletonConfigurable):
    # this changing of these two values will be captured by traitlets
    color_value = traitlets.Any()
    depth_value = traitlets.Any()

    def __init__(self):
        super(Camera, self).__init__()

        # configure the color and depth sensor
        self.pipeline = rs.pipeline()
        self.configuration = rs.config()

        # set resolution for the color camera
        self.color_width = 640
        self.color_height = 480
        self.color_fps = 30
        self.configuration.enable_stream(rs.stream.color, self.color_width, self.color_height, rs.format.bgr8,
                                         self.color_fps)

        # set resolution for the depth camera
        self.depth_width = 640
        self.depth_height = 480
        self.depth_fps = 30
        self.configuration.enable_stream(rs.stream.depth, self.depth_width, self.depth_height, rs.format.z16,
                                         self.depth_fps)

        self.stop_flag = False

        # start capture the first depth and color image
        self.pipeline.start(self.configuration)
        self.pipeline_started = True
        frames = self.pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()
        image = np.asanyarray(color_frame.get_data())
        self.color_value = image

        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_JET)
        self.depth_value = depth_colormap

    def _capture_frames(self):
        while (self.stop_flag == False):
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            image = np.asanyarray(color_frame.get_data())
            self.color_value = image

            depth_frame = frames.get_depth_frame()
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.05), cv2.COLORMAP_JET)
            self.depth_value = depth_colormap
            self.depth_image = depth_image

    def start(self):
        if not hasattr(self, 'thread') or not self.thread.isAlive():
            self.thread = threading.Thread(target=self._capture_frames)
            self.thread.start()

    def stop(self):
        self.stop_flag = True
        if hasattr(self, 'thread'):
            self.thread.join()
        if self.pipeline_started:
            self.pipeline.stop()