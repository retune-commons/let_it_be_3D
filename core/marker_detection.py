from typing import List, Tuple, Optional, Union, Dict
from abc import ABC, abstractmethod
from pathlib import Path
import yaml
import imageio.v3 as iio
import matplotlib.pyplot as plt
import pandas as pd

from .utils import Coordinates
"""
import datetime
import itertools as it
import aniposelib as ap_lib
import cv2
"""


class MarkerDetection(ABC):
    def __init__(
        self,
        object_to_analyse: Path,
        output_directory: Path,
        marker_detection_directory: Optional[Path] = None,
    ):
        self.object_to_analyse = object_to_analyse
        self.output_directory = output_directory
        if type(marker_detection_directory) != None:
            self.marker_detection_directory = marker_detection_directory

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
        )
        return df


class TemplateMatching(MarkerDetection):
    def analyze_objects(self):
        # self.object_to_analyse, self.output_directory, self.marker_detection_directory
        pass


class ManualAnnotation(MarkerDetection):
    def analyze_objects(self, filename):    
        with open(self.marker_detection_directory, "r") as ymlfile:
                list_of_labels = yaml.load(ymlfile, Loader=yaml.SafeLoader)

        frames_annotated = []
        for frame in iio.imread(self.object_to_analyse):
            frame_annotated = {}
            plt.figure(figsize=(10, 5))
            plt.imshow(frame)
            plt.show()
            for label in list_of_labels:
                likelihood = 1
                y = input(f"{label}: y_or_row_index\nIf you want to skip this marker, enter x!")
                if y == "x":
                    likelihood, x, y = 0, 0, 0
                else:
                    x = input(f"{label}: x_or_column_index")
                frame_annotated[label] = (int(x), int(y), likelihood)
            frames_annotated.append(frame_annotated)

        #deeplabcut df from dict        
        df = 
        #save as hdf
        df.to_h5(filename)