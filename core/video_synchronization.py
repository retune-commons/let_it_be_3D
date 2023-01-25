from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union, Dict
import random
import multiprocessing as mp
import time

from tqdm.auto import tqdm as TQDM
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import imageio as iio
import ffmpeg
import pandas as pd

from .video_metadata import VideoMetadata
from .utils import Coordinates
from .marker_detection import DeeplabcutInterface, ManualAnnotation
from .plotting import Alignment_Plot_Individual, LED_Marker_Plot


class TimeseriesTemplate(ABC):
    @property
    @abstractmethod
    def template_attribute_string(self) -> str:
        pass

    def adjust_template_timeseries_to_fps(self, fps: int) -> List[np.ndarray]:
        template_timeseries = getattr(self, self.template_attribute_string)
        fps_adjusted_templates = []
        framerate = fps / 1000
        max_frames = int(template_timeseries.shape[0] * framerate)
        max_offset = 1000 // fps
        for offset_in_ms in range(max_offset):
            image_timestamps = np.linspace(
                0 + offset_in_ms,
                template_timeseries.shape[0] + offset_in_ms,
                max_frames,
                dtype="int",
            )
            while image_timestamps[-1] >= template_timeseries.shape[0]:
                image_timestamps = image_timestamps[:-1]
            adjusted_template = template_timeseries[image_timestamps].copy()
            fps_adjusted_templates.append((adjusted_template, offset_in_ms))
        return fps_adjusted_templates


class MotifTemplate(TimeseriesTemplate):
    @property
    def template_attribute_string(self) -> str:
        return "template_timeseries"

    def __init__(
        self,
        led_on_time_in_ms: int,
        on_off_period_length_in_ms: int,
        motif_duration_in_ms: int,
    ):
        self.led_on_time_in_ms = led_on_time_in_ms
        self.on_off_period_length_in_ms = on_off_period_length_in_ms
        self.motif_duration_in_ms = motif_duration_in_ms
        self.template_timeseries = self._compute_template_timeseries()

    def _compute_template_timeseries(self) -> np.ndarray:
        led_on_off_period = np.zeros((self.on_off_period_length_in_ms), dtype="float")
        led_on_off_period[1 : self.led_on_time_in_ms + 1] = 1
        full_repetitions = self.motif_duration_in_ms // self.on_off_period_length_in_ms
        remaining_ms = self.motif_duration_in_ms % self.on_off_period_length_in_ms
        motif_template = np.concatenate([led_on_off_period] * (full_repetitions + 1))
        adjusted_end_index = (
            self.on_off_period_length_in_ms * full_repetitions + remaining_ms
        )
        return motif_template[:adjusted_end_index]


class MultiMotifTemplate(TimeseriesTemplate):
    @property
    def template_attribute_string(self) -> str:
        return "multi_motif_template"

    def __init__(self) -> None:
        self.motif_templates = []

    def add_motif_template(self, motif_template: MotifTemplate) -> None:
        self.motif_templates.append(motif_template)
        self.multi_motif_template = self._update_session_template()

    def _update_session_template(self) -> np.ndarray:
        individual_motif_template_timeseries = [
            elem.template_timeseries for elem in self.motif_templates
        ]
        return np.concatenate(individual_motif_template_timeseries)


class Synchronizer(ABC):
    @property
    # differnet thresholds for different patterns!
    def alignment_threshold(self) -> int:
        return 100

    @property
    # different thresholds for different patterns!
    def box_size(self) -> int:
        return 15

    @property
    @abstractmethod
    def target_fps(self) -> int:
        pass

    @abstractmethod
    def _adjust_video_to_target_fps_and_run_marker_detection(
        self, target_fps: int, start_idx: int, offset: float
    ) -> Path:
        # call corresponding private method that takes care of two things:
        #   - marker detection
        #   - dropping / adding frames and potentially interpolation
        # the private methods also determine the order, for instance:
        #   - downsampling: drop frames first & then detect on the reduced data
        #   - upsampling: detect markers first and then add missing frames (potentially interpolate markers)
        # eventually returns the filepath where the relevant synchronized output was saved, which can be:
        #   - filepath to the synchronized charucoboard video (.mp4 file)
        #   - filepath to the DLC output of the detected markers (.h5 file)
        pass

    def __init__(
        self, video_metadata: VideoMetadata, use_gpu: bool, output_directory: Path
    ) -> None:
        self.video_metadata = video_metadata
        self.use_gpu = use_gpu
        self.output_directory = output_directory

    def run_synchronization(
        self, synchronize_only: bool, test_mode: bool = False
    ) -> Tuple[Path, bool]:
        self.template_blinking_motif = self._construct_template_motif(
            blinking_patterns_metadata=self.video_metadata.led_pattern
        )

        synchronized_video_filepath = self._construct_video_filepath()
        if (not test_mode) or (test_mode and not synchronized_video_filepath.exists()):
            i = 0
            while True:
                if i == 3:
                    """
                    self.video_metadata.led_extraction_type = "manual"
                    led_center_coordinates = self._get_LED_center_coordinates()
                    """
                    print(
                        "Could not synchronize the video. \n"
                        "Make sure, that you chose the right synchronization pattern, \n"
                        "that the LED is visible during the pattern\n"
                        "and that you chose a proper alignment threshold!"
                    )
                    self.video_metadata.exclusion_state = "exclude"
                    return None, None
                elif i < 3:
                    led_center_coordinates = self._get_LED_center_coordinates()
                else:
                    print(
                        "Could not synchronize the video. \n"
                        "Make sure, that you chose the right synchronization pattern, \n"
                        "that the LED is visible during the pattern\n"
                        "and that you chose a proper alignment threshold!"
                    )
                    self.video_metadata.exclusion_state = "exclude"
                    return None, None

                self.led_timeseries = self._extract_led_pixel_intensities(
                    led_center_coords=led_center_coordinates
                )

                (
                    offset_adjusted_start_idx,
                    remaining_offset,
                    alignment_error,
                ) = self._find_best_match_of_template(
                    template=self.template_blinking_motif, start_time=0, end_time=len(self.led_timeseries)*0.4
                )
                # make synchronization adaptable: (if below threshold: repeat/continue anyways/manual input)
                if alignment_error > self.alignment_threshold:
                    print("possibly bad synchronization!")
                break
                # i += 1

            self.led_detection = LED_Marker_Plot(
                image=iio.v3.imread(self.video_metadata.filepath, index=0),
                led_center_coordinates=led_center_coordinates,
                box_size=self.box_size,
                video_metadata=self.video_metadata,
                output_directory=self.output_directory,
            )

            self.led_timeseries_for_cross_video_validation = (
                self._adjust_led_timeseries_for_cross_validation(
                    start_idx=offset_adjusted_start_idx, offset=remaining_offset
                )
            )
        else:
            offset_adjusted_start_idx = None
            remaining_offset = None
            
        (
            marker_detection_filepath,
            synchronized_video_filepath,
        ) = self._adjust_video_to_target_fps_and_run_marker_detection(
            target_fps=self.target_fps,
            start_idx=offset_adjusted_start_idx,
            offset=remaining_offset,
            synchronize_only=synchronize_only,
            test_mode=test_mode,
        )
        self.video_metadata.framenum_synchronized = iio.v2.get_reader(
            synchronized_video_filepath
        ).count_frames()
        self.video_metadata.duration_synchronized = (
            self.video_metadata.framenum_synchronized / self.video_metadata.target_fps
        )
        return marker_detection_filepath, synchronized_video_filepath

    def _construct_template_motif(
        self, blinking_patterns_metadata: Dict
    ) -> Union[MotifTemplate, MultiMotifTemplate]:
        motif_templates = []
        for pattern_idx, parameters in blinking_patterns_metadata.items():
            motif_templates.append(
                MotifTemplate(
                    led_on_time_in_ms=parameters["led_on_time_in_ms"],
                    on_off_period_length_in_ms=parameters["on_off_period_length_in_ms"],
                    motif_duration_in_ms=parameters["motif_duration_in_ms"],
                )
            )
        if len(motif_templates) < 1:
            raise ValueError(
                "Could not construct a blinking pattern template. Please validate your config files!"
            )
        elif len(motif_templates) == 1:
            template_motif = motif_templates[0]
        else:
            template_motif = MultiMotifTemplate()
            for template in motif_templates:
                template_motif.add_motif_template(motif_template=template)
        return template_motif

    def _get_LED_center_coordinates(self) -> Coordinates:
        if self.video_metadata.led_extraction_type == "DLC":
            temp_folder = self.output_directory.joinpath("temp")
            Path.mkdir(temp_folder, exist_ok=True)
            video_filepath_out = temp_folder.joinpath(
                f"{self.video_metadata.recording_date}_{self.video_metadata.cam_id}_LED_detection_samples.mp4"
            )
            config_filepath = self.video_metadata.led_extraction_filepath
            if self.video_metadata.charuco_video:
                dlc_filepath_out = temp_folder.joinpath(
                    f"{self.video_metadata.recording_date}_{self.video_metadata.cam_id}_LED_detection_predictions.h5"
                )
            else:
                dlc_filepath_out = temp_folder.joinpath(
                    f"{self.video_metadata.mouse_id}_{self.video_metadata.recording_date}_{self.video_metadata.paradigm}_{self.video_metadata.cam_id}.h5"
                )
            
            num_frames_to_pick = 100
            if num_frames_to_pick > self.video_metadata.framenum:
                num_frames_to_pick = int(self.video_metadata.framenum/2)
            sample_frame_idxs = random.sample(
                range(self.video_metadata.framenum),
                num_frames_to_pick,
            )

            selected_frames = []
            for idx in sample_frame_idxs:
                selected_frames.append(
                    iio.v3.imread(self.video_metadata.filepath, index=idx)
                )
            video_array = np.asarray(selected_frames)
            iio.v3.imwrite(
                str(video_filepath_out), video_array, fps=1, macro_block_size=1
            )

            dlc_interface = DeeplabcutInterface(
                object_to_analyse=str(video_filepath_out),
                output_directory=temp_folder,
                marker_detection_directory=config_filepath,
            )
            dlc_ending = dlc_interface.analyze_objects(filtering=False)
            dlc_created_h5file = temp_folder.joinpath(
                video_filepath_out.stem + dlc_ending + ".h5"
            )
            dlc_created_picklefile = temp_folder.joinpath(
                video_filepath_out.stem + dlc_ending + "_meta.pickle"
            )
            dlc_created_h5file.rename(dlc_filepath_out)
            df = pd.read_hdf(dlc_filepath_out)
            x_key, y_key, likelihood_key = (
                [key for key in df.keys() if "LED5" in key and "x" in key],
                [key for key in df.keys() if "LED5" in key and "y" in key],
                [key for key in df.keys() if "LED5" in key and "likelihood" in key],
            )
            x = int(df.loc[df[likelihood_key].idxmax(), x_key].values)
            y = int(df.loc[df[likelihood_key].idxmax(), y_key].values)
            video_filepath_out.unlink()
            dlc_filepath_out.unlink()
            dlc_created_picklefile.unlink()
            temp_folder.rmdir()
            return Coordinates(y_or_row=y, x_or_column=x)

        elif self.video_metadata.led_extraction_type == "manual":
            temp_folder = self.output_directory.joinpath("temp")
            Path.mkdir(temp_folder, exist_ok=True)

            config_filepath = self.video_metadata.led_extraction_filepath
            if self.video_metadata.charuco_video:
                manual_filepath_out = temp_folder.joinpath(
                    f"{self.video_metadata.recording_date}_{self.video_metadata.cam_id}_LED_detection_predictions.h5"
                )
            else:
                manual_filepath_out = temp_folder.joinpath(
                    f"{self.video_metadata.mouse_id}_{self.video_metadata.recording_date}_{self.video_metadata.paradigm}_{self.video_metadata.cam_id}.h5"
                )

            manual_interface = ManualAnnotation(
                object_to_analyse=self.video_metadata.filepath,
                output_directory=self.output_directory,
                marker_detection_directory=config_filepath,
            )
            manual_interface.analyze_objects(
                filepath=manual_filepath_out, labels=["LED5"], only_first_frame=True
            )

            df = pd.read_hdf(manual_filepath_out)
            x_key, y_key = (
                [key for key in df.keys() if "LED5" in key and "x" in key],
                [key for key in df.keys() if "LED5" in key and "y" in key],
            )
            x = int(df.loc[0, x_key].values)
            y = int(df.loc[0, y_key].values)

            manual_filepath_out.unlink()
            return Coordinates(y_or_row=y, x_or_column=x)
        else:
            print("Template Matching is not yet implemented!")

    def _extract_led_pixel_intensities(
        self, led_center_coords: Coordinates
    ) -> np.ndarray:
        box_row_indices = self._get_start_end_indices_from_center_coord_and_length(
            center_px=led_center_coords.row, length=self.box_size
        )
        box_col_indices = self._get_start_end_indices_from_center_coord_and_length(
            center_px=led_center_coords.column, length=self.box_size
        )
        mean_pixel_intensities = self._calculate_mean_pixel_intensities(
            box_row_indices=box_row_indices, box_col_indices=box_col_indices
        )
        return np.asarray(mean_pixel_intensities)

    def _get_start_end_indices_from_center_coord_and_length(
        self, center_px: int, length: int
    ) -> Tuple[int, int]:
        start_index = center_px - (length // 2)
        end_index = center_px + (length - (length // 2))
        return start_index, end_index

    def _calculate_mean_pixel_intensities(
        self, box_row_indices: Tuple[int, int], box_col_indices: Tuple[int, int]
    ) -> List[float]:
        mean_pixel_intensities = []
        for frame in iio.v3.imiter(self.video_metadata.filepath):
            box_mean_intensity = frame[
                box_row_indices[0] : box_row_indices[1],
                box_col_indices[0] : box_col_indices[1],
            ].mean()
            mean_pixel_intensities.append(box_mean_intensity)
        return mean_pixel_intensities

    def _find_best_match_of_template(
        self,
        template: Union[MotifTemplate, MultiMotifTemplate],
        start_time: int = 0,
        end_time: int = -1,
    ) -> Tuple[int, float]:
        adjusted_motif_timeseries = template.adjust_template_timeseries_to_fps(
            fps=self.video_metadata.fps
        )
        start_frame_idx = self._get_frame_index_closest_to_time(time=start_time)
        end_frame_idx = self._get_frame_index_closest_to_time(time=end_time)
        (
            best_match_start_frame_idx,
            best_match_offset,
            alignment_error,
        ) = self._get_start_index_and_offset_of_best_match(
            adjusted_templates=adjusted_motif_timeseries,
            start_frame_idx=start_frame_idx,
            end_frame_idx=end_frame_idx,
        )
        offset_adjusted_start_idx, remaining_offset = self._adjust_start_idx_and_offset(
            start_frame_idx=best_match_start_frame_idx,
            offset=best_match_offset,
            fps=self.video_metadata.fps,
        )

        self.synchronization_individual = Alignment_Plot_Individual(
            template=adjusted_motif_timeseries[best_match_offset][0],
            led_timeseries=self.led_timeseries[best_match_start_frame_idx:],
            video_metadata=self.video_metadata,
            output_directory=self.output_directory,
            led_box_size=self.box_size,
        )

        return offset_adjusted_start_idx, remaining_offset, alignment_error

    def _get_frame_index_closest_to_time(self, time: int) -> int:
        time_error_message = (
            f"The specified time: {time} is invalid! Both times have to be an integer "
            "larger than -1, where -1 represents the very last timestamp in the video "
            "and every other integer (e.g.: 1000) the time in ms. Please be aware, that "
            '"start_time" has to be larger than "end_time" (with end_time == -1 as only '
            "exception) and must be smaller or equal to the total video recording time."
        )
        if time == -1:
            return -1
        else:
            if time < 0:
                raise ValueError(time_error_message)
            framerate = 1000 / self.video_metadata.fps  # adapt
            closest_frame_idx = round(time / framerate)
            if closest_frame_idx >= self.led_timeseries.shape[0]:
                raise ValueError(time_error_message)
            return closest_frame_idx

    def _get_start_index_and_offset_of_best_match(
        self,
        adjusted_templates: List[Tuple[np.ndarray, int]],
        start_frame_idx: int,
        end_frame_idx: int,
    ) -> Tuple[int, int]:
        # ToDo - check if error is within range to automatically accept as good match, otherwise report file & save plots for inspection
        lowest_sum_of_squared_error_per_template = []
        for template_timeseries, _ in adjusted_templates:
            alignment_results = self._run_alignment(
                query=template_timeseries,
                subject=self.led_timeseries[start_frame_idx:end_frame_idx],
            )
            lowest_sum_of_squared_error_per_template.append(alignment_results.min())
        lowest_sum_of_squared_error_per_template = np.asarray(
            lowest_sum_of_squared_error_per_template
        )
        best_matching_template_index = int(
            lowest_sum_of_squared_error_per_template.argmin()
        )
        best_alignment_results = self._run_alignment(
            query=adjusted_templates[best_matching_template_index][0],
            subject=self.led_timeseries[start_frame_idx:end_frame_idx],
        )
        start_index = best_alignment_results.argmin()

        return start_index, best_matching_template_index, best_alignment_results.min()

    def _adjust_start_idx_and_offset(
        self, start_frame_idx: int, offset: int, fps: int
    ) -> Tuple[int, float]:
        n_frames_to_add = round(offset / fps)
        adjusted_start_frame_idx = start_frame_idx + n_frames_to_add
        original_ms_per_frame = self._get_ms_interval_per_frame(fps=fps)
        remaining_offset = offset - n_frames_to_add * original_ms_per_frame
        return adjusted_start_frame_idx, remaining_offset

    def _run_alignment(self, query: np.ndarray, subject: np.ndarray) -> np.ndarray:
        if self.use_gpu:
            alignment_results = self._run_rapid_aligner(query=query, subject=subject)
        else:
            alignment_results = self._run_cpu_aligner(query=query, subject=subject)
        return alignment_results

    def _run_rapid_aligner(self, query: np.ndarray, subject: np.ndarray) -> np.ndarray:
        import sys

        # Todo: implement in project_config.yaml and video_metadata objects
        sys.path.append("/home/ds/GitHub_repos/rapidAligner/")
        import cupy as cp
        import rapidAligner as ra

        subject_timeseries_gpu = cp.asarray(subject)
        query_timeseries_gpu = cp.asarray(query)
        alignment_results = ra.ED.zdist(
            query_timeseries_gpu, subject_timeseries_gpu, mode="fft"
        )
        return alignment_results.get()

    def _run_cpu_aligner(self, query: np.ndarray, subject: np.ndarray) -> np.ndarray:
        # same as rapidAligner, just using numpy instead of cupy
        return self._fft_zdist(q=query, s=subject, epsilon=1e-6)

    def _fft_zdist(self, q: np.ndarray, s: np.ndarray, epsilon: float):
        alignment, kahan = 10_000, 0
        m, q = len(q), self._znorm(q, epsilon)
        n = (len(s) + alignment - 1) // alignment * alignment
        is_ = np.zeros(n, dtype=s.dtype)
        is_[: len(s)] = s
        delta = n - len(s)
        x, y = self._cumsum(is_, kahan), self._cumsum(is_**2, kahan)
        x = x[+m:] - x[:-m]
        y = y[+m:] - y[:-m]
        z = np.sqrt(np.maximum(y / m - np.square(x / m), 0))
        e = np.zeros(n, dtype=q.dtype)
        e[:m] = q
        r = np.fft.irfft(np.fft.rfft(e).conj() * np.fft.rfft(is_), n=n)
        np.seterr(divide="ignore")  # toggle output
        np.seterr(invalid="ignore")  # toggle output
        f = np.where(z > 0, 2 * (m - r[: -m + 1] / z), m * np.ones_like(z))
        np.seterr(divide="warn")
        np.seterr(invalid="warn")
        return f[: len(s) - m + 1]

    def _cumsum(self, x, kahan=0):
        assert isinstance(kahan, int) and kahan >= 0
        y = np.empty(len(x) + 1, dtype=x.dtype)
        y[0] = 0
        np.cumsum(x, out=y[1:])
        if kahan:
            r = x - np.diff(y)
            if np.max(np.abs(r)):
                y += cumsum(r, kahan - 1)
        return y

    def _znorm(self, x, epsilon):
        return (x - np.mean(x)) / max(np.std(x, ddof=0), epsilon)

    def _adjust_led_timeseries_for_cross_validation(
        self, start_idx: int, offset: float
    ) -> np.ndarray:
        adjusted_led_timeseries = self.led_timeseries[start_idx:].copy()
        if self.video_metadata.fps != self.video_metadata.target_fps:
            adjusted_led_timeseries = self._downsample_led_timeseries(
                timeseries=adjusted_led_timeseries, offset=offset
            )
        return adjusted_led_timeseries

    def _downsample_led_timeseries(
        self, timeseries: np.ndarray, offset: float
    ) -> np.ndarray:
        n_frames_after_downsampling = self._compute_fps_adjusted_frame_count(
            original_n_frames=timeseries.shape[0],
            original_fps=self.video_metadata.fps,
            target_fps=self.video_metadata.target_fps,
        )
        original_timestamps = self._compute_timestamps(
            n_frames=timeseries.shape[0], fps=self.video_metadata.fps, offset=offset
        )
        target_timestamps = self._compute_timestamps(
            n_frames=n_frames_after_downsampling, fps=self.video_metadata.target_fps
        )
        frame_idxs_best_matching_timestamps = (
            self._find_frame_idxs_closest_to_target_timestamps(
                target_timestamps=target_timestamps,
                original_timestamps=original_timestamps,
            )
        )
        return timeseries[frame_idxs_best_matching_timestamps]

    def _compute_fps_adjusted_frame_count(
        self, original_n_frames: int, original_fps: int, target_fps: int
    ) -> int:
        target_ms_per_frame = self._get_ms_interval_per_frame(fps=target_fps)
        original_ms_per_frame = self._get_ms_interval_per_frame(fps=original_fps)
        return int((original_n_frames * original_ms_per_frame) / target_ms_per_frame)

    def _compute_timestamps(
        self, n_frames: int, fps: int, offset: float = 0.0
    ) -> np.ndarray:
        ms_per_frame = self._get_ms_interval_per_frame(fps=fps)
        timestamps = np.arange(n_frames * ms_per_frame, step=ms_per_frame)
        return timestamps + offset

    def _find_closest_timestamp_index(
        self, original_timestamps: np.ndarray, timestamp: float
    ) -> int:
        return np.abs(original_timestamps - timestamp).argmin()

    def _find_frame_idxs_closest_to_target_timestamps(
        self, target_timestamps: np.ndarray, original_timestamps: np.ndarray
    ) -> List[int]:
        frame_indices_closest_to_target_timestamps = []
        for timestamp in target_timestamps:
            closest_frame_index = self._find_closest_timestamp_index(
                original_timestamps=original_timestamps, timestamp=timestamp
            )
            frame_indices_closest_to_target_timestamps.append(closest_frame_index)
        return frame_indices_closest_to_target_timestamps

    def _get_ms_interval_per_frame(self, fps: int) -> float:
        return 1000 / fps

    def _downsample_video(
        self,
        start_idx: int,
        offset: float,
        target_fps: int = 30,
        test_mode: bool = False,
    ) -> Path:
        if test_mode:
            filepath_downsampled_video = self._construct_video_filepath()
            if filepath_downsampled_video.exists():
                return filepath_downsampled_video
        frame_idxs_to_sample = self._get_sampling_frame_idxs(
            start_idx=start_idx, offset=offset, target_fps=target_fps
        )
        sampling_frame_idxs_per_part = self._split_into_ram_digestable_parts(
            idxs_of_frames_to_sample=frame_idxs_to_sample,
            max_ram_digestible_frames=self.video_metadata.max_ram_digestible_frames,
        )
        if len(frame_idxs_to_sample) > 1:
            filepaths_all_video_parts = (
                self._initiate_iterative_writing_of_individual_video_parts(
                    frame_idxs_to_sample=sampling_frame_idxs_per_part,
                )
            )
            filepath_downsampled_video = (
                self._concatenate_individual_video_parts_on_disk(
                    filepaths_of_video_parts=filepaths_all_video_parts
                )
            )
            self._delete_individual_video_parts(
                filepaths_of_video_parts=filepaths_all_video_parts
            )
        else:
            filepath_downsampled_video = self._write_video_to_disk(
                idxs_of_frames_to_sample=frame_idxs_to_sample[0],
                target_fps=target_fps,
            )
        return filepath_downsampled_video

    def _get_sampling_frame_idxs(
        self, start_idx: int, offset: float, target_fps: int
    ) -> List[int]:
        original_n_frames = self.led_timeseries[start_idx:].shape[0]
        n_frames_after_downsampling = self._compute_fps_adjusted_frame_count(
            original_n_frames=original_n_frames,
            original_fps=self.video_metadata.fps,
            target_fps=target_fps,
        )
        original_timestamps = self._compute_timestamps(
            n_frames=original_n_frames, fps=self.video_metadata.fps, offset=offset
        )
        target_timestamps = self._compute_timestamps(
            n_frames=n_frames_after_downsampling, fps=target_fps
        )
        frame_idxs_best_matching_timestamps = (
            self._find_frame_idxs_closest_to_target_timestamps(
                target_timestamps=target_timestamps,
                original_timestamps=original_timestamps,
            )
        )
        sampling_frame_idxs = self._adjust_frame_idxs_for_synchronization_shift(
            unadjusted_frame_idxs=frame_idxs_best_matching_timestamps,
            start_idx=start_idx,
        )
        return sampling_frame_idxs

    def _adjust_frame_idxs_for_synchronization_shift(
        self, unadjusted_frame_idxs: List[int], start_idx: int
    ) -> List[int]:
        adjusted_frame_idxs = np.asarray(unadjusted_frame_idxs) + start_idx
        return list(adjusted_frame_idxs)

    def _split_into_ram_digestable_parts(
        self, idxs_of_frames_to_sample: List[int], max_ram_digestible_frames: int
    ) -> List[List[int]]:
        frame_idxs_to_sample = []
        while len(idxs_of_frames_to_sample) > max_ram_digestible_frames:
            frame_idxs_to_sample.append(idxs_of_frames_to_sample[:max_ram_digestible_frames])
            idxs_of_frames_to_sample = idxs_of_frames_to_sample[max_ram_digestible_frames:]
        frame_idxs_to_sample.append(idxs_of_frames_to_sample)
        return frame_idxs_to_sample

    def _initiate_iterative_writing_of_individual_video_parts(
        self, frame_idxs_to_sample: List[List[int]]) -> List[Path]:
        
        if self.video_metadata.max_cpu_cores_to_pool > 1:
            available_cpus = mp.cpu_count()
            if available_cpus > self.video_metadata.max_cpu_cores_to_pool:
                num_processes = self.video_metadata.max_cpu_cores_to_pool
            else:
                num_processes = available_cpus
            with mp.Pool(num_processes) as p:
                filepaths_to_all_video_parts = p.map(self._multiprocessing_function, enumerate(frame_idxs_to_sample))

        else:
            filepaths_to_all_video_parts = []
            for idx, idxs_of_frames_to_sample in enumerate(frame_idxs_to_sample):
                part_id = str(idx).zfill(3)
                filepath_video_part = self._write_video_to_disk(
                    idxs_of_frames_to_sample=idxs_of_frames_to_sample,
                    target_fps=self.target_fps,
                    part_id=part_id,
                )
                filepaths_to_all_video_parts.append(filepath_video_part)

        return filepaths_to_all_video_parts
    
    def _multiprocessing_function(self, idx_and_idxs_of_frames_to_sample: Tuple)->None:
        idx = idx_and_idxs_of_frames_to_sample[0]
        idxs_of_frames_to_sample = idx_and_idxs_of_frames_to_sample[1]
        part_id = str(idx).zfill(3)
        filepath_video_part = self._write_video_to_disk(
            idxs_of_frames_to_sample=idxs_of_frames_to_sample,
            target_fps=self.target_fps,
            part_id=part_id,
        )
        return filepath_video_part

    def _write_video_to_disk(
        self,
        idxs_of_frames_to_sample: List[int],
        target_fps: int,
        part_id: Optional[int] = None,
    ) -> Path:
        selected_frames = []
        for i, frame in enumerate(iio.v3.imiter(self.video_metadata.filepath)):
            if i > idxs_of_frames_to_sample[-1]:
                break
            if i in idxs_of_frames_to_sample:
                selected_frames.append(frame)
        video_array = np.asarray(selected_frames)
        filepath_out = self._construct_video_filepath(part_id=part_id)
        iio.v3.imwrite(filepath_out, video_array, fps=target_fps, macro_block_size=1)
        self.synchronized_object_filepath = filepath_out
        return filepath_out

    def _construct_video_filepath(self, part_id: Optional[int] = None) -> Path:
        if self.video_metadata.charuco_video:
            if part_id == None:
                filepath = self.output_directory.joinpath(
                    f"{self.video_metadata.recording_date}_{self.video_metadata.cam_id}_synchronized.mp4"
                )
            else:
                filepath = self.output_directory.joinpath(
                    f"{self.video_metadata.recording_date}_{self.video_metadata.cam_id}_synchronized_part_{part_id}.mp4"
                )

        else:
            if part_id == None:
                filepath = self.output_directory.joinpath(
                    f"{self.video_metadata.mouse_id}_{self.video_metadata.recording_date}_{self.video_metadata.paradigm}_{self.video_metadata.cam_id}_synchronized.mp4"
                )
            else:
                filepath = self.output_directory.joinpath(
                    f"{self.video_metadata.mouse_id}_{self.video_metadata.recording_date}_{self.video_metadata.paradigm}_{self.video_metadata.cam_id}_synchronized_part_{part_id}.mp4"
                )
        return filepath

    def _concatenate_individual_video_parts_on_disk(
        self, filepaths_of_video_parts: List[Path]
    ) -> Path:
        video_part_streams = [
            ffmpeg.input(str(filepath)) for filepath in filepaths_of_video_parts
        ]
        filepath_concatenated_video = self._construct_video_filepath(part_id=None)
        if len(video_part_streams) >= 2:
            concatenated_video = ffmpeg.concat(
                video_part_streams[0], video_part_streams[1]
            )
            if len(video_part_streams) >= 3:
                for part_stream in video_part_streams[2:]:
                    concatenated_video = ffmpeg.concat(concatenated_video, part_stream)
            output_stream = ffmpeg.output(
                concatenated_video, filename=str(filepath_concatenated_video)
            )
        else:
            output_stream = ffmpeg.output(
                video_part_streams[0], filename=str(filepath_concatenated_video))
        output_stream.run(overwrite_output=True, quiet=True)
        return filepath_concatenated_video

    def _delete_individual_video_parts(
        self, filepaths_of_video_parts: List[Path]
    ) -> None:
        for filepath in filepaths_of_video_parts:
            filepath.unlink()


class CharucoVideoSynchronizer(Synchronizer):
    @property
    def target_fps(self) -> int:
        return self.video_metadata.target_fps

    def _adjust_video_to_target_fps_and_run_marker_detection(
        self,
        target_fps: int,
        start_idx: int,
        offset: float,
        test_mode: bool,
        synchronize_only: bool = True,
    ) -> Tuple[Path]:
        return None, self._downsample_video(
            start_idx=start_idx,
            offset=offset,
            target_fps=self.target_fps,
            test_mode=test_mode,
        )


class RecordingVideoSynchronizer(Synchronizer):
    def _run_deep_lab_cut_for_marker_detection(
        self, video_filepath: Path, test_mode: bool = False
    ) -> Path:
        output_filepath = self._create_h5_filepath()

        if not test_mode and  not output_filepath.exists():
            config_filepath = self.video_metadata.processing_filepath
            dlc_interface = DeeplabcutInterface(
                object_to_analyse=str(video_filepath),
                output_directory=self.output_directory,
                marker_detection_directory=config_filepath,
            )
            h5_file = dlc_interface.analyze_objects(filtering=True)
            created_filepath = self.output_directory.joinpath(
                video_filepath.stem + h5_file
            )

            try:
                Path(str(created_filepath) + ".h5").rename(Path(str(output_filepath) + ".h5"))
            except:
                pass
            try:
                Path(str(created_filepath) + "_filtered.h5").rename(Path(str(output_filepath) + "_filtered.h5"))
            except:
                pass

        return Path(str(output_filepath) + ".h5")

    def _run_manual_marker_detection(
        self, video_filepath: Path, test_mode: bool = False
    ) -> Path:
        output_filepath = self._create_h5_filepath()

        if not test_mode and not output_filepath.exists():
            inp = input(
                "Are you sure, you want to overwrite the existing file and label a new one? (y/n)"
            )
            if inp == "y":
                config_filepath = self.video_metadata.processing_filepath
                manual_interface = ManualAnnotation(
                    object_to_analyse=video_filepath,
                    output_directory=self.output_directory,
                    marker_detection_directory=config_filepath,
                )
                manual_interface.analyze_objects(filepath= str(output_filepath) + ".h5")

        return output_filepath

    def _create_h5_filepath(self) -> Path:
        h5_filepath = self.output_directory.joinpath(
            f"{self.video_metadata.mouse_id}_{self.video_metadata.recording_date}_{self.video_metadata.paradigm}_{self.video_metadata.cam_id}"
        )
        return h5_filepath


class RecordingVideoDownSynchronizer(RecordingVideoSynchronizer):
    @property
    def target_fps(self) -> int:
        return self.video_metadata.target_fps

    def _adjust_video_to_target_fps_and_run_marker_detection(
        self,
        target_fps: int,
        start_idx: int,
        offset: float,
        synchronize_only: bool,
        test_mode: bool,
    ) -> Tuple[Path]:
        downsampled_video_filepath = self._downsample_video(
            start_idx=start_idx,
            offset=offset,
            target_fps=self.target_fps,
            test_mode=test_mode,
        )
        if not synchronize_only:
            if self.video_metadata.processing_type == "DLC":
                detected_markers_filepath = self._run_deep_lab_cut_for_marker_detection(
                    video_filepath=downsampled_video_filepath, test_mode=test_mode
                )
            elif self.video_metadata.processing_type == "manual":
                detected_markers_filepath = self._run_manual_marker_detection(
                    video_filepath=downsampled_video_filepath, test_mode=test_mode
                )
            else:
                print("TemplateMatching is not yet implemented!")
            return detected_markers_filepath, downsampled_video_filepath
        else:
            return None, downsampled_video_filepath


class RecordingVideoUpSynchronizer(RecordingVideoSynchronizer):
    @property
    def target_fps(self) -> int:
        return self.video_metadata.target_fps

    def _adjust_video_to_target_fps_and_run_marker_detection(self):
        pass
