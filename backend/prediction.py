# Represents a computed prediction object for Aurora
from dataclasses import dataclass
from typing import Optional
import os

import numpy as np

@dataclass
class Prediction(object):

    temperature: np.ndarray  # Temperature data at 100m height
    u_wind_100m: np.ndarray   # U component of wind at 100m height
    v_wind_100m: np.ndarray   # V component of wind at


    def to_file(self, filename):
        """
        Save the prediction data to a file.
        """
        np.savez(os.path.expanduser(filename),
            temperature=self.temperature,
            u_wind_100m=self.u_wind_100m,
            v_wind_100m=self.v_wind_100m
        )
    
    def from_file(filename) -> Optional['Prediction']:
        """
        Load the prediction data from a file.
        """
        try:
            data = np.load(os.path.expanduser(filename))
            return Prediction(
                temperature=data['temperature'],
                u_wind_100m=data['u_wind_100m'],
                v_wind_100m=data['v_wind_100m']
            )
        except FileNotFoundError:
            return None