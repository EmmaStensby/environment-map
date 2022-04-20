import math
import random

class Controller:
    def __init__(self, amplitude=None, period=None, phase=None):
        # Amplitude 
        if amplitude is not None:
            self.amplitude = amplitude
        else:
            self.amplitude = random.uniform(0,1)*math.pi

        # Period
        if period is not None:
            self.period = period/8.0
        else:
            self.period = random.uniform(0,1)/8.0

        # Phase
        if phase is not None:
            self.phase = phase*2*math.pi
        else:
            self.phase = random.uniform(0,1)*2*math.pi

    def update(self, time):
        return self.amplitude * math.sin(time*self.period + self.phase)
