from typing import List, Tuple, Optional, Union, Dict
from abc import ABC, abstractmethod
from pathlib import Path
import sys
import io
import warnings

import yaml
import imageio.v3 as iio
import matplotlib.pyplot as plt
import pandas as pd


from .utils import construct_dlc_output_style_df_from_manual_marker_coords, convert_to_path, read_config


class MarkerDetection(ABC):
    def __init__(
        self,
        object_to_analyse: Path,
        output_directory: Path,
        marker_detection_directory: Optional[Path] = None,
    ):
        self.object_to_analyse = convert_to_path(object_to_analyse)
        self.output_directory = convert_to_path(output_directory)
        if type(marker_detection_directory) != None:
            self.marker_detection_directory = convert_to_path(marker_detection_directory)

    @abstractmethod
    def analyze_objects():
        pass


class DeeplabcutInterface(MarkerDetection):
    def analyze_objects(self, filepath: Path, filtering: bool = False, use_gpu: str=""):
        convert_to_path(filepath)
        if use_gpu == "prevent":
            import tensorflow.compat.v1 as tf
            sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
        elif use_gpu == "low":
            import tensorflow.compat.v1 as tf
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        elif use_gpu == "full":
            import tensorflow.compat.v1 as tf
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        
        #mute deeplabcut
        old_stdout = sys.stdout
        text_trap = io.StringIO()
        sys.stdout = text_trap
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            import deeplabcut as dlc

            dlc_ending = dlc.analyze_videos(
                config=str(self.marker_detection_directory),
                videos=[str(self.object_to_analyse)],
                destfolder=str(self.output_directory),
            )
            unfiltered_filepath = self.output_directory.joinpath(self.object_to_analyse.stem + dlc_ending + '.h5')
            unfiltered_filepath.rename(filepath)
            if filtering:
                dlc.post_processing.filtering.filterpredictions(
                    config=str(self.marker_detection_directory), video=str(self.object_to_analyse), save_as_csv=False
                )
                filtered_filepath = self.output_directory.joinpath(unfiltered_filepath.stem + "_filtered.h5")
                filtered_filepath.rename(filepath.stem + "_filtered.h5")
            
        #unmute 
        sys.stdout = old_stdout

        return filepath


class ManualAnnotation(MarkerDetection):
    def analyze_objects(
        self,
        filepath: Path,
        labels: Optional[List[str]] = None,
        only_first_frame: bool = False,
    ) -> Path:
        if labels == None:
            ground_truth_config = read_config(self.marker_detection_directory)
            list_of_labels = ground_truth_config['unique_ids']
        else:
            list_of_labels = labels

        frames_annotated = {}

        for label in list_of_labels:
            frames_annotated[label] = {"x": [], "y": [], "likelihood": []}
        for frame in iio.imiter(self.object_to_analyse):
            plt.close()
            fig, ax = plt.subplots(figsize=(10, 7))
            plt.imshow(frame)
            y_lim = frame.shape[0]
            x_lim = frame.shape[1]
            x_ticks = [i for i in range(x_lim) if i % 10 == 0]
            y_ticks = [i for i in range(y_lim) if i % 10 == 0]
            x_labels = [i if i % 50 == 0 else " " for i in x_ticks]
            y_labels = [i if i % 50 == 0 else " " for i in y_ticks]
            ax.set_xticks(x_ticks, labels=x_labels)
            ax.set_yticks(y_ticks, labels=y_labels)
            plt.grid(visible=True, c="black", alpha=0.25)
            plt.show()

            for label in list_of_labels:
                likelihood = 1
                y = input(
                    f"{label}: y_or_row_index\nIf you want to skip this marker, enter x!"
                )
                if y == "x":
                    likelihood, x, y = 0, 0, 0
                else:
                    x = input(f"{label}: x_or_column_index")
                try:
                    int(x)
                    int(y)
                except ValueError:
                    likelihood, x, y = 0, 0, 0
                frames_annotated[label]["x"].append(int(x))
                frames_annotated[label]["y"].append(int(y))
                frames_annotated[label]["likelihood"].append(likelihood)
            if only_first_frame:
                break

        df = construct_dlc_output_style_df_from_manual_marker_coords(
            manual_test_position_marker_coords_pred=frames_annotated
        )
        df.to_hdf(filepath, "df")
        return filepath


class TemplateMatching(MarkerDetection):
    def analyze_objects(self):
        # self.object_to_analyse, self.output_directory, self.marker_detection_directory
        pass
