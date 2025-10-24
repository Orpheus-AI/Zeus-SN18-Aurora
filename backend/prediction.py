# Represents a computed prediction object for Aurora
from dataclasses import dataclass
from typing import Optional
import os

import numpy as np

@dataclass
class Prediction(object):

    temperature: np.ndarray  # Temperature data at 2m height

    def to_file(self, filename):
        """
        Save the prediction data to a file.
        """
        np.savez(os.path.expanduser(filename),
            temperature=self.temperature,
        )
    
    def from_file(filename) -> Optional['Prediction']:
        """
        Load the prediction data from a file.
        """
        try:
            data = np.load(os.path.expanduser(filename))
            return Prediction(
                temperature=data['temperature'],
            )
        except FileNotFoundError:
            return None

    def get(self, variable: str) -> Optional[np.ndarray]:
        """
        Get a specific variable from the prediction.
        """
        if variable == "2m_temperature":
            return self.temperature
        raise NotImplementedError(f"Variable {variable} not implemented.")