import traitlets
import time
from traitlets.config.configurable import SingletonConfigurable

class Robot(SingletonConfigurable):
    
    front_left_motor = None
    front_right_motor = None
    back_left_motor = None
    back_right_motor = None

    def __init__(self, *args, **kwargs):
        super(Robot, self).__init__(*args, **kwargs)
        pass
    def set_motors(self, front_left_speed, front_right_speed, back_left_speed, back_right_speed):
        pass
    def forward(self, speed=1.0):
        pass

    def backward(self, speed=1.0):
        pass

    def left(self, speed=1.0):
        pass

    def right(self, speed=1.0):
        pass

    def stop(self):
        pass
    
    def forward_left(self, speed=1.0):
        pass
            
    def forward_right(self, speed=1.0):
        pass

    def backward_left(self, speed=1.0):
        pass
            
    def backward_right(self, speed=1.0):
        pass
