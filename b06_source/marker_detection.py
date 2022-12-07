from typing import List, Tuple, Optional, Union, Dict
from abc import ABC, abstractmethod
from pathlib import Path

"""
import datetime
import itertools as it
import imageio.v3 as iio
import numpy as np
import aniposelib as ap_lib
import cv2
import matplotlib.pyplot as plt
"""


class MarkerDetection(ABC):
    def __init__(
        self,
        object_to_analyse: Path,
        output_directory: Path,
        marker_detection_directory: Optional[Path] = None,
        dynamic: Optional[bool] = None,
    ):
        self.object_to_analyse = object_to_analyse
        self.output_directory = output_directory
        try:
            self.marker_detection_directory = marker_detection_directory
            try:
                self.dynamic = (
                    dynamic,
                    0.5,
                    10,
                )  # Tuple that DLC needs. 0.5 and 10 are default threshold & margin
            except ValueError:
                pass
        except ValueError:
            pass

    @abstractmethod
    def analyze_objects():
        pass


class DeeplabcutInterface(MarkerDetection):
    def analyze_objects(self):
        import deeplabcut as dlc

        df = dlc.analyze_videos(
            config=self.marker_detection_directory,
            videos=[self.object_to_analyse],
            destfolder=self.output_directory,
            dynamic=self.dynamic,
        )
        return df


class TemplateMatching(MarkerDetection):
    def analyze_objects(self):
        # self.object_to_analyse, self.output_directory, self.marker_detection_directory
        pass


class ManualAnnotation(MarkerDetection):
    def analyze_objects(self):
        # self.object_to_analyse, self.output_directory
        pass
