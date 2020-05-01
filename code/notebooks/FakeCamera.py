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


        # set resolution for the color camera
        self.color_width = 640
        self.color_height = 480
        self.color_fps = 30

        # set resolution for the depth camera
        self.depth_width = 640
        self.depth_height = 480
        self.depth_fps = 30

        self.stop_flag = False

        # start capture the first depth and color image

        self.fake_image = np.zeros((480,640, 3))
        self.fake_image2 = np.zeros((480,640, 3))
        self.color_value = self.fake_image

        self.depth_value = self.fake_image2

    def _capture_frames(self):
        while (self.stop_flag == False):
            self.color_value = self.fake_image

            self.depth_value = self.fake_image2
            self.depth_image = self.fake_image2

    def start(self):
        if not hasattr(self, 'thread') or not self.thread.isAlive():
            self.thread = threading.Thread(target=self._capture_frames)
            self.thread.start()

    def stop(self):
        self.stop_flag = True
        if hasattr(self, 'thread'):
            self.thread.join()
