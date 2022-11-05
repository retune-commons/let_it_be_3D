from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union, Dict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from b06_source.video_metadata import VideoMetadata
from b06_source.utils import Coordinates


class Plotting(ABC):
    
    def _save(self, filepath: str):
        plt.savefig(filepath, dpi=400)
    
    @abstractmethod
    def plot(self):
        pass
    
    @abstractmethod
    def _create_filename(self):
        pass
    
    def _zscore(self, array: np.ndarray) -> np.ndarray:
        return (array-np.mean(array))/np.std(array, ddof=0) 
    
    
class Alignment_Plot_Individual(Plotting):
    
    def __init__(self, template: np.ndarray, led_timeseries: np.ndarray, video_metadata: VideoMetadata, output_directory: Path) -> None:
        self.template = template
        self.led_timeseries = led_timeseries
        self.video_metadata = video_metadata
        self.output_directory = output_directory
        
    def plot(self, save_to_disk: bool=False) -> None:
        filepath = self._create_filename()
        self._create_plot(filepath = filepath, save_to_disk = save_to_disk)
        
    def _create_filename(self) -> str:
        if self.video_metadata.charuco_video:
            filename = f'{self.video_metadata.recording_date}_{self.video_metadata.cam_id}_charuco_synchronization_individual'
        else:
            filename = f'{self.video_metadata.mouse_id}_{self.video_metadata.recording_date}_{self.video_metadata.paradigm}_{self.video_metadata.cam_id}_synchronization_individual'
        filepath = self.output_directory.joinpath(filename)
        return str(filepath)
        
    def _create_plot(self, filepath: str, save_to_disk: bool) -> None:
        end_idx = self.template.shape[0]
        fig = plt.figure(figsize=(9, 6), facecolor='white')
        plt.plot(self._zscore(array = self.led_timeseries[:end_idx]))
        plt.plot(self._zscore(array = self.template))
        plt.title(f'{self.video_metadata.cam_id}')
        if save_to_disk:
            self._save(filepath = filepath)
        plt.show()
    
        
        
class Alignment_Plot_Crossvalidation(Plotting):
    
    def __init__(self, template: np.ndarray, led_timeseries: Dict, metadata: Dict, output_directory: Path):
        self.template = template
        self.led_timeseries = led_timeseries
        self.metadata = metadata
        self.output_directory = output_directory
        
    def plot(self, save_to_disk: bool=True) -> None:
        filepath = self._create_filename()
        self._create_plot(filepath = filepath, save_to_disk = save_to_disk)
    
    def _create_filename(self) -> str:
        if self.metadata['charuco_video']:
            filename = f'{self.metadata["recording_date"]}_charuco_synchronization_crossvalidation'
        else:
            filename = f'{self.metadata["mouse_id"]}_{self.metadata["recording_date"]}_{self.metadata["paradigm"]}_synchronization_crossvalidation'
        filepath = self.output_directory.joinpath(filename)
        return str(filepath)
    
    def _create_plot(self, filepath: str, save_to_disk: bool):
        fig = plt.figure(figsize=(9, 6), facecolor='white')
        end_idx = self.template.shape[0]
        for label in self.led_timeseries.keys():
            led_timeseries = self.led_timeseries[label]
            plt.plot(self._zscore(array = led_timeseries[:end_idx]), label = label)
        plt.plot(self._zscore(array = self.template), c='black', label = 'Template') 
        plt.legend()
        if save_to_disk:
            self._save(filepath = filepath)
        plt.show()

        
        
#class intrinsic calibrations


class LED_Marker_Plot(Plotting):
    def __init__(self, image: np.ndarray, led_center_coordinates: Coordinates, box_size: int, video_metadata: VideoMetadata, output_directory: Path) -> None:
        self.image = image
        self.led_center_coordinates = led_center_coordinates
        self.box_size = box_size
        self.video_metadata = video_metadata
        self.output_directory = output_directory
        
    def plot(self, save_to_disk: bool=True) -> None:
        filepath = self._create_filename()
        self._create_plot(filepath = filepath, save_to_disk = save_to_disk)
        
    def _create_filename(self):
        if self.video_metadata.charuco_video:
            filename = f'{self.video_metadata.recording_date}_{self.video_metadata.cam_id}_charuco_LED_marker'
        else:
            filename = f'{self.video_metadata.mouse_id}_{self.video_metadata.recording_date}_{self.video_metadata.paradigm}_{self.video_metadata.cam_id}_LED_marker'
        filepath = self.output_directory.joinpath(filename)
        return filepath

    def _create_plot(self, save_to_disk: bool):
        fig = plt.figure()
        plt.imshow(self.image)
        plt.scatter(self.led_center_coordinates.x, self.led_center_coordinates.y)
        #plt.plot: box
        if save_to_disk:
            self._save(filepath = filepath)
        plt.show()
        
        
