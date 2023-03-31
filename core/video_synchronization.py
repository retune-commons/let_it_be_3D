import multiprocessing as mp
import pickle
import random
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any

import ffmpeg
import imageio as iio
import numpy as np
import pandas as pd
import scipy

from .marker_detection import DeeplabcutInterface, ManualAnnotation
from .plotting import AlignmentPlotIndividual, LEDMarkerPlot
from .utils import Coordinates
from .video_metadata import VideoMetadata


class TimeseriesTemplate(ABC):
    @property
    @abstractmethod
    def template_attribute_string(self) -> str:
        pass

    def adjust_template_timeseries_to_fps(self, fps: int) -> List[Tuple[Any, int]]:
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
        led_on_off_period = np.zeros(self.on_off_period_length_in_ms, dtype="float")
        led_on_off_period[1: self.led_on_time_in_ms + 1] = 1
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
        self.multi_motif_template = None
        self.motif_templates = []

    def add_motif_template(self, motif_template: MotifTemplate) -> None:
        self.motif_templates.append(motif_template)
        self.multi_motif_template = self._update_session_template()

    def _update_session_template(self) -> np.ndarray:
        individual_motif_template_timeseries = [
            elem.template_timeseries for elem in self.motif_templates
        ]
        return np.concatenate(individual_motif_template_timeseries)


def _find_closest_timestamp_index(
        original_timestamps: np.ndarray, timestamp: float
) -> int:
    return np.abs(original_timestamps - timestamp).argmin()


def _find_frame_idxs_closest_to_target_timestamps(
        target_timestamps: np.ndarray, original_timestamps: np.ndarray
) -> List[int]:
    frame_indices_closest_to_target_timestamps = []
    for timestamp in target_timestamps:
        closest_frame_index = _find_closest_timestamp_index(
            original_timestamps=original_timestamps, timestamp=timestamp
        )
        frame_indices_closest_to_target_timestamps.append(closest_frame_index)
    return frame_indices_closest_to_target_timestamps


def _get_ms_interval_per_frame(fps: int) -> float:
    return 1000 / fps


def _adjust_frame_idxs_for_synchronization_shift(
        unadjusted_frame_idxs: List[int], start_idx: int
) -> List[int]:
    adjusted_frame_idxs = np.asarray(unadjusted_frame_idxs) + start_idx
    return list(adjusted_frame_idxs)


def _compute_fps_adjusted_frame_count(
        original_n_frames: int, original_fps: int, target_fps: int
) -> int:
    target_ms_per_frame = _get_ms_interval_per_frame(fps=target_fps)
    original_ms_per_frame = _get_ms_interval_per_frame(fps=original_fps)
    return int((original_n_frames * original_ms_per_frame) / target_ms_per_frame)


def _compute_timestamps(
        n_frames: int, fps: int, offset: float = 0.0
) -> np.ndarray:
    ms_per_frame = _get_ms_interval_per_frame(fps=fps)
    timestamps = np.arange(n_frames * ms_per_frame, step=ms_per_frame)
    return timestamps + offset


def _znorm(x, epsilon):
    return (x - np.mean(x)) / max(np.std(x, ddof=0), epsilon)


def _adjust_start_idx_and_offset(
        start_frame_idx: int, offset: int, fps: int
) -> Tuple[int, float]:
    n_frames_to_add = round(offset / fps)
    adjusted_start_frame_idx = start_frame_idx + n_frames_to_add
    original_ms_per_frame = _get_ms_interval_per_frame(fps=fps)
    remaining_offset = offset - n_frames_to_add * original_ms_per_frame
    return adjusted_start_frame_idx, remaining_offset


def _get_start_end_indices_from_center_coord_and_length(
        center_px: int, length: int
) -> Tuple[int, int]:
    start_index = center_px - (length // 2)
    end_index = center_px + (length - (length // 2))
    return start_index, end_index


def _construct_template_motif(
        blinking_patterns_metadata: Dict
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


def _load_synchro(filepath: Path) -> Tuple[Coordinates, Any, Any, Any]:
    with open(filepath, "rb") as file:
        synchro_object = pickle.load(file)
    return Coordinates(synchro_object["led_center_coordinates"][0], synchro_object["led_center_coordinates"][1]), \
        synchro_object["offset_adjusted_start_idx"], synchro_object["remaining_offset"], synchro_object[
        "alignment_error"]


def _cumsum(x, kahan=0):
    assert isinstance(kahan, int) and kahan >= 0
    y = np.empty(len(x) + 1, dtype=x.dtype)
    y[0] = 0
    np.cumsum(x, out=y[1:])
    if kahan:
        r = x - np.diff(y)
        if np.max(np.abs(r)):
            y += np.cumsum(r, kahan - 1)
    return y


def _fft_zdist(q: np.ndarray, s: np.ndarray, epsilon: float):
    alignment, kahan = 10_000, 0
    m, q = len(q), _znorm(q, epsilon)
    n = (len(s) + alignment - 1) // alignment * alignment
    is_ = np.zeros(n, dtype=s.dtype)
    is_[: len(s)] = s
    delta = n - len(s)
    x, y = _cumsum(is_, kahan), _cumsum(is_ ** 2, kahan)
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


def _run_cpu_aligner(query: np.ndarray, subject: np.ndarray) -> np.ndarray:
    # same as rapidAligner, just using numpy instead of cupy
    return _fft_zdist(q=query, s=subject, epsilon=1e-6)


def _split_into_ram_digestable_parts(idxs_of_frames_to_sample: List[int], max_ram_digestible_frames: int
                                     ) -> List[List[int]]:
    frame_idxs_to_sample = []
    while len(idxs_of_frames_to_sample) > max_ram_digestible_frames:
        frame_idxs_to_sample.append(
            idxs_of_frames_to_sample[:max_ram_digestible_frames]
        )
        idxs_of_frames_to_sample = idxs_of_frames_to_sample[
                                   max_ram_digestible_frames:
                                   ]
    frame_idxs_to_sample.append(idxs_of_frames_to_sample)
    return frame_idxs_to_sample


def _delete_individual_video_parts(filepaths_of_video_parts: List[Path]
                                   ) -> None:
    for filepath in filepaths_of_video_parts:
        filepath.unlink()


class Synchronizer(ABC):
    @property
    def box_size(self) -> int:
        return 15

    @property
    def target_fps(self) -> int:
        return self.video_metadata.target_fps

    @abstractmethod
    def _adjust_video_to_target_fps_and_run_marker_detection(
            self, target_fps: int, start_idx: int, offset: float, test_mode: bool, synchronize_only: bool
    ) -> Tuple[Optional[Path], Optional[Path]]:
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

    @abstractmethod
    def _create_h5_filepath(self, tag: str = "_rawfps_unsynchronized", filtered: bool = False) -> Path:
        pass

    def __init__(
            self,
            video_metadata: VideoMetadata,
            rapid_aligner_path: Path,
            output_directory: Path,
            synchro_metadata: Dict,
            use_gpu: str = "",
    ) -> None:
        self.led_timeseries_for_cross_video_validation = None
        self.led_detection = None
        self.led_timeseries = None
        self.template_blinking_motif = None
        self.video_metadata = video_metadata
        if rapid_aligner_path.name != "":
            self.use_rapid_aligner = True
            self.rapid_aligner_path = rapid_aligner_path
        else:
            self.use_rapid_aligner = False
        self.output_directory = output_directory
        self.use_gpu = use_gpu
        self.synchro_metadata = synchro_metadata

    def run_synchronization(
            self, synchronize_only: bool, test_mode: bool = False
    ) -> Tuple[Optional[Path], Optional[Path]]:

        self.template_blinking_motif = _construct_template_motif(
            blinking_patterns_metadata=self.video_metadata.led_pattern)

        preexisting_output_file = self._get_preexisting_output_filepath()
        synchro_file = self._get_synchro_filepath()

        if test_mode and preexisting_output_file.exists():
            marker_detection_filepath, synchronized_video_filepath = None, None
            if preexisting_output_file.suffix == ".h5":
                marker_detection_filepath, synchronized_video_filepath = preexisting_output_file, None
            elif preexisting_output_file.suffix == ".mp4":
                synchronized_video_filepath, marker_detection_filepath = preexisting_output_file, None
        else:
            if test_mode and synchro_file.exists():
                led_center_coordinates, offset_adjusted_start_idx, remaining_offset, alignment_error = _load_synchro(
                    filepath=synchro_file)
            else:
                led_center_coordinates = self._get_led_center_coordinates()
                self.led_timeseries = self._extract_led_pixel_intensities(led_center_coords=led_center_coordinates)
                offset_adjusted_start_idx, remaining_offset, alignment_error = self._find_best_match_of_template(
                    template=self.template_blinking_motif, start_time=self.synchro_metadata["start_pattern_match_ms"],
                    end_time=self.synchro_metadata["end_pattern_match_ms"])

                if alignment_error > self.synchro_metadata["synchro_error_threshold"]:
                    led_center_coordinates, offset_adjusted_start_idx, remaining_offset, alignment_error = self._handle_synchro_fails()

                self.led_detection = LEDMarkerPlot(
                    image=iio.v3.imread(self.video_metadata.filepath, index=0),
                    led_center_coordinates=led_center_coordinates,
                    box_size=self.box_size,
                    video_metadata=self.video_metadata,
                    output_directory=self.output_directory,
                )
                self.led_timeseries_for_cross_video_validation = \
                    self._adjust_led_timeseries_for_cross_validation(start_idx=offset_adjusted_start_idx,
                                                                     offset=remaining_offset)
                self._save_synchro(filepath=synchro_file,
                                   led_center_coordinates=led_center_coordinates,
                                   offset_adjusted_start_idx=offset_adjusted_start_idx,
                                   remaining_offset=remaining_offset,
                                   alignment_error=alignment_error)

            marker_detection_filepath, synchronized_video_filepath = \
                self._adjust_video_to_target_fps_and_run_marker_detection(target_fps=self.target_fps,
                                                                          start_idx=offset_adjusted_start_idx,
                                                                          offset=remaining_offset,
                                                                          synchronize_only=synchronize_only,
                                                                          test_mode=test_mode)
        self.video_metadata.framenum_synchronized, self.video_metadata.duration_synchronized = \
            self._get_framenumber_of_synchronized_files(synchronize_only=synchronize_only,
                                                        marker_detection_filepath=marker_detection_filepath,
                                                        synchronized_video_filepath=synchronized_video_filepath)

        return marker_detection_filepath, synchronized_video_filepath

    def _save_synchro(self, filepath: Path, led_center_coordinates: Coordinates, offset_adjusted_start_idx: int,
                      remaining_offset: int, alignment_error: float) -> None:
        time = datetime.now().strftime("%Y%m%d%H%M%S")
        synchro_dict = {"filepath": str(self.video_metadata.filepath),
                        "fps": self.video_metadata.fps,
                        "led_center_coordinates": (led_center_coordinates.y, led_center_coordinates.x),
                        "offset_adjusted_start_idx": offset_adjusted_start_idx,
                        "remaining_offset": remaining_offset,
                        "alignment_error": alignment_error,
                        "time": time,
                        "scorer": self.video_metadata.led_extraction_filepath,
                        "synchro_marker": self.synchro_metadata["synchro_marker"],
                        "method": self.video_metadata.led_extraction_type,
                        "pattern": self.video_metadata.led_pattern}
        with open(filepath, "wb") as file:
            pickle.dump(synchro_dict, file)

    def _get_framenumber_of_synchronized_files(self, synchronize_only: bool, marker_detection_filepath: Path,
                                               synchronized_video_filepath: Path) -> Tuple[Any, Any]:
        if synchronize_only:
            framenum_synchronized = iio.v2.get_reader(synchronized_video_filepath).count_frames()
        else:
            framenum_synchronized = pd.read_hdf(marker_detection_filepath).shape[0]
        duration_synchronized = (framenum_synchronized / self.video_metadata.target_fps)
        print(f"{self.video_metadata.cam_id} Frames after synchronization: {framenum_synchronized}")

        return framenum_synchronized, duration_synchronized

    def _handle_synchro_fails(self) -> Tuple[Coordinates, Union[int, Any], Union[float, Any], Union[int, Any]]:
        if self.synchro_metadata["handle_synchro_fails"] == "repeat":
            led_center_coordinates = self._get_led_center_coordinates()
            offset_adjusted_start_idx, remaining_offset, alignment_error = self._find_best_match_of_template(
                template=self.template_blinking_motif, start_time=self.synchro_metadata["start_pattern_match_ms"],
                end_time=self.synchro_metadata["end_pattern_match_ms"])
        elif self.synchro_metadata["handle_synchro_fails"] == "default":
            alignment_error = 0
            led_center_coordinates = Coordinates(0, 0)
            default_offset = self.synchro_metadata["default_offset_ms"]
            default_start_idx = self._get_frame_index_closest_to_time(time=default_offset)
            offset_adjusted_start_idx, remaining_offset = _adjust_start_idx_and_offset(
                start_frame_idx=default_start_idx, offset=default_offset, fps=self.video_metadata.fps)
        elif self.synchro_metadata["handle_synchro_fails"] == "manual":
            self.video_metadata.led_extraction_type = "manual"
            led_center_coordinates = self._get_led_center_coordinates()
            offset_adjusted_start_idx, remaining_offset, alignment_error = self._find_best_match_of_template(
                template=self.template_blinking_motif, start_time=self.synchro_metadata["start_pattern_match_ms"],
                end_time=self.synchro_metadata["end_pattern_match_ms"])
        elif self.synchro_metadata["handle_synchro_fails"] == "error":
            raise ValueError(
                "Could not synchronize the video. \n"
                "Make sure, that you chose the right synchronization pattern, \n"
                "that the LED is visible during the pattern\n"
                "and that you chose a proper alignment threshold!")
        else:
            raise ValueError(
                "project_config key handle_synchro_fails has to be in ['error', 'manual', 'repeat', 'default']")
        return led_center_coordinates, offset_adjusted_start_idx, remaining_offset, alignment_error

    def _get_preexisting_output_filepath(self) -> Path:
        output_file = None
        if type(self) == CharucoVideoSynchronizer:
            output_file = self._construct_video_filepath()
        elif type(self) == RecordingVideoUpSynchronizer:
            output_file = self._create_h5_filepath(tag=f"_upsampled{self.target_fps}fps_synchronized",
                                                   filtered=self.synchro_metadata['use_2D_filter'])
        elif type(self) == RecordingVideoDownSynchronizer:
            output_file = self._create_h5_filepath(tag=f"_downsampled{self.target_fps}fps_synchronized",
                                                   filtered=self.synchro_metadata['use_2D_filter'])
        return output_file

    def _get_synchro_filepath(self) -> Path:
        return self.output_directory.joinpath(
            f"synchro_{self.video_metadata.recording_date}_{self.video_metadata.cam_id}.p")

    def _get_led_center_coordinates(self) -> Coordinates:
        temp_folder = self.output_directory.joinpath("temp")
        Path.mkdir(temp_folder, exist_ok=True)

        if self.video_metadata.led_extraction_type == "DLC":
            video_filepath_out = temp_folder.joinpath(
                f"{self.video_metadata.recording_date}_{self.video_metadata.cam_id}_LED_detection_samples.mp4"
            )
            if self.video_metadata.charuco_video:
                dlc_filepath_out = temp_folder.joinpath(
                    f"{self.video_metadata.recording_date}_{self.video_metadata.cam_id}_LED_detection_predictions.h5"
                )
            else:
                dlc_filepath_out = temp_folder.joinpath(
                    f"{self.video_metadata.mouse_id}_{self.video_metadata.recording_date}_{self.video_metadata.paradigm}_{self.video_metadata.cam_id}.h5"
                )

            num_frames_to_pick = self.synchro_metadata['num_frames_to_pick']
            if num_frames_to_pick > self.video_metadata.framenum:
                num_frames_to_pick = int(self.video_metadata.framenum / 2)
            sample_frame_idxs = random.sample(range(self.video_metadata.framenum), num_frames_to_pick, )
            selected_frames = []
            for idx in sample_frame_idxs:
                selected_frames.append(
                    iio.v3.imread(self.video_metadata.filepath, index=idx)
                )
            video_array = np.asarray(selected_frames)
            iio.v3.imwrite(str(video_filepath_out), video_array, fps=1, macro_block_size=1)

            dlc_interface = DeeplabcutInterface(
                object_to_analyse=video_filepath_out,
                output_directory=temp_folder,
                marker_detection_directory=self.video_metadata.led_extraction_filepath,
            )
            dlc_filepath_out = dlc_interface.analyze_objects(
                filtering=False, use_gpu=self.use_gpu, filepath=dlc_filepath_out
            )
            for file in temp_folder.iterdir():
                if file.suffix == ".pickle":
                    dlc_created_picklefile = file
                    dlc_created_picklefile.unlink()

            df = pd.read_hdf(dlc_filepath_out)
            x_key, y_key, likelihood_key = (
                [key for key in df.keys() if self.synchro_metadata["synchro_marker"] in key and "x" in key],
                [key for key in df.keys() if self.synchro_metadata["synchro_marker"] in key and "y" in key],
                [key for key in df.keys() if self.synchro_metadata["synchro_marker"] in key and "likelihood" in key],
            )
            x = int(df.loc[df[likelihood_key].idxmax(), x_key].values)
            y = int(df.loc[df[likelihood_key].idxmax(), y_key].values)
            video_filepath_out.unlink()
            dlc_filepath_out.unlink()

        elif self.video_metadata.led_extraction_type == "manual":
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
                filepath=manual_filepath_out, labels=[self.synchro_metadata["synchro_marker"]], only_first_frame=True
            )

            df = pd.read_hdf(manual_filepath_out)
            x_key, y_key = (
                [key for key in df.keys() if self.synchro_metadata["synchro_marker"] in key and "x" in key],
                [key for key in df.keys() if self.synchro_metadata["synchro_marker"] in key and "y" in key],
            )
            x = int(df.loc[0, x_key].values)
            y = int(df.loc[0, y_key].values)

            manual_filepath_out.unlink()
        else:
            raise ValueError("For LED extraction only DLC and manual are supported!")
        temp_folder.rmdir()
        return Coordinates(y_or_row=y, x_or_column=x)

    def _extract_led_pixel_intensities(
            self, led_center_coords: Coordinates
    ) -> np.ndarray:
        box_row_indices = _get_start_end_indices_from_center_coord_and_length(
            center_px=led_center_coords.row, length=self.box_size
        )
        box_col_indices = _get_start_end_indices_from_center_coord_and_length(
            center_px=led_center_coords.column, length=self.box_size
        )
        mean_pixel_intensities = self._calculate_mean_pixel_intensities(
            box_row_indices=box_row_indices, box_col_indices=box_col_indices
        )
        return np.asarray(mean_pixel_intensities)

    def _calculate_mean_pixel_intensities(
            self, box_row_indices: Tuple[int, int], box_col_indices: Tuple[int, int]
    ) -> List[float]:
        mean_pixel_intensities = []
        for frame in iio.v3.imiter(self.video_metadata.filepath):
            box_mean_intensity = frame[
                                 box_row_indices[0]: box_row_indices[1],
                                 box_col_indices[0]: box_col_indices[1],
                                 ].mean()
            mean_pixel_intensities.append(box_mean_intensity)
        return mean_pixel_intensities

    def _find_best_match_of_template(
            self,
            template: Union[MotifTemplate, MultiMotifTemplate],
            start_time: int = 0,
            end_time: int = -1,
    ) -> Tuple[int, float, Any]:
        adjusted_motif_timeseries = template.adjust_template_timeseries_to_fps(
            fps=self.video_metadata.fps
        )
        start_frame_idx = self._get_frame_index_closest_to_time(time=start_time)
        try:
            end_frame_idx = self._get_frame_index_closest_to_time(time=end_time)
        except ValueError:
            end_frame_idx = -1  # if the given end_time is larger than the length of the corresponding video, its set to the last frame of the video
        (
            best_match_start_frame_idx,
            best_match_offset,
            alignment_error,
        ) = self._get_start_index_and_offset_of_best_match(
            adjusted_templates=adjusted_motif_timeseries,
            start_frame_idx=start_frame_idx,
            end_frame_idx=end_frame_idx,
        )
        offset_adjusted_start_idx, remaining_offset = _adjust_start_idx_and_offset(
            start_frame_idx=best_match_start_frame_idx,
            offset=best_match_offset,
            fps=self.video_metadata.fps,
        )

        self.synchronization_individual = AlignmentPlotIndividual(
            template=adjusted_motif_timeseries[best_match_offset][0],
            led_timeseries=self.led_timeseries[best_match_start_frame_idx:],
            video_metadata=self.video_metadata,
            output_directory=self.output_directory,
            led_box_size=self.box_size,
            alignment_error=alignment_error,
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
            framerate = 1000 / self.video_metadata.fps
            closest_frame_idx = round(time / framerate)
            if closest_frame_idx >= self.led_timeseries.shape[0]:
                raise ValueError(time_error_message)
            return closest_frame_idx

    def _get_start_index_and_offset_of_best_match(
            self,
            adjusted_templates: List[Tuple[np.ndarray, int]],
            start_frame_idx: int,
            end_frame_idx: int,
    ) -> Tuple[Any, int, Any]:
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

    def _run_alignment(self, query: np.ndarray, subject: np.ndarray) -> np.ndarray:
        if self.use_rapid_aligner:
            alignment_results = self._run_rapid_aligner(query=query, subject=subject)
        else:
            alignment_results = _run_cpu_aligner(query=query, subject=subject)
        return alignment_results

    def _run_rapid_aligner(self, query: np.ndarray, subject: np.ndarray) -> np.ndarray:
        import sys

        sys.path.append(str(self.rapid_aligner_path))
        import cupy as cp
        import rapidAligner as ra

        subject_timeseries_gpu = cp.asarray(subject)
        query_timeseries_gpu = cp.asarray(query)
        alignment_results = ra.ED.zdist(
            query_timeseries_gpu, subject_timeseries_gpu, mode="fft"
        )
        return alignment_results.get()

    def _adjust_led_timeseries_for_cross_validation(
            self, start_idx: int, offset: float
    ) -> np.ndarray:
        adjusted_led_timeseries = self.led_timeseries[start_idx:].copy()
        if self.video_metadata.fps > self.video_metadata.target_fps:
            adjusted_led_timeseries = self._downsample_led_timeseries(
                timeseries=adjusted_led_timeseries, offset=offset
            )
        else:
            adjusted_led_timeseries = self._upsample_led_timeseries(
                timeseries=adjusted_led_timeseries, offset=offset
            )
        return adjusted_led_timeseries

    def _downsample_led_timeseries(
            self, timeseries: np.ndarray, offset: float
    ) -> np.ndarray:
        n_frames_after_downsampling = _compute_fps_adjusted_frame_count(
            original_n_frames=timeseries.shape[0],
            original_fps=self.video_metadata.fps,
            target_fps=self.video_metadata.target_fps,
        )
        original_timestamps = _compute_timestamps(
            n_frames=timeseries.shape[0], fps=self.video_metadata.fps, offset=offset
        )
        target_timestamps = _compute_timestamps(
            n_frames=n_frames_after_downsampling, fps=self.video_metadata.target_fps
        )
        frame_idxs_best_matching_timestamps = (
            _find_frame_idxs_closest_to_target_timestamps(
                target_timestamps=target_timestamps,
                original_timestamps=original_timestamps,
            )
        )
        return timeseries[frame_idxs_best_matching_timestamps]

    def _upsample_led_timeseries(
            self, timeseries: np.ndarray, offset: float
    ) -> np.ndarray:
        len_frame_in_ms = 1 / self.video_metadata.fps * 1000
        len_targetframe_in_ms = 1 / self.target_fps * 1000
        index_in_ms = np.arange(
            0, timeseries.shape[0] * len_frame_in_ms, len_frame_in_ms
        )
        new_indices = np.arange(
            offset, (timeseries.shape[0] - 1) * len_frame_in_ms, len_targetframe_in_ms
        )
        d = scipy.interpolate.interp1d(index_in_ms, timeseries)
        upsampled_timeseries = d(new_indices)
        return upsampled_timeseries

    def _downsample_video(
            self,
            start_idx: int,
            offset: float,
            target_fps: int = 30,
            test_mode: bool = False,
    ) -> Path:
        if test_mode:
            preexisting_filepath_downsampled_video = self._construct_video_filepath()
            if preexisting_filepath_downsampled_video.exists():
                return preexisting_filepath_downsampled_video
        frame_idxs_to_sample = self._get_sampling_frame_idxs(start_idx=start_idx, offset=offset, target_fps=target_fps)
        sampling_frame_idxs_per_part = _split_into_ram_digestable_parts(
            idxs_of_frames_to_sample=frame_idxs_to_sample,
            max_ram_digestible_frames=self.video_metadata.max_ram_digestible_frames,
        )
        if len(frame_idxs_to_sample) > 1:
            filepaths_all_video_parts = self._initiate_iterative_writing_of_individual_video_parts(
                frame_idxs_to_sample=sampling_frame_idxs_per_part)
            filepath_downsampled_video = self._concatenate_individual_video_parts_on_disk(
                filepaths_of_video_parts=filepaths_all_video_parts)
            _delete_individual_video_parts(filepaths_of_video_parts=filepaths_all_video_parts)
        else:
            filepath_downsampled_video = self._write_video_to_disk(idxs_of_frames_to_sample=frame_idxs_to_sample[0],
                                                                   target_fps=target_fps, )
        return filepath_downsampled_video

    def _get_sampling_frame_idxs(
            self, start_idx: int, offset: float, target_fps: int
    ) -> List[int]:
        original_n_frames = self.video_metadata.framenum - start_idx
        n_frames_after_downsampling = _compute_fps_adjusted_frame_count(
            original_n_frames=original_n_frames,
            original_fps=self.video_metadata.fps,
            target_fps=target_fps,
        )
        original_timestamps = _compute_timestamps(
            n_frames=original_n_frames, fps=self.video_metadata.fps, offset=offset
        )
        target_timestamps = _compute_timestamps(
            n_frames=n_frames_after_downsampling, fps=target_fps
        )
        frame_idxs_best_matching_timestamps = (
            _find_frame_idxs_closest_to_target_timestamps(
                target_timestamps=target_timestamps,
                original_timestamps=original_timestamps,
            )
        )
        sampling_frame_idxs = _adjust_frame_idxs_for_synchronization_shift(
            unadjusted_frame_idxs=frame_idxs_best_matching_timestamps,
            start_idx=start_idx,
        )
        return sampling_frame_idxs

    def _initiate_iterative_writing_of_individual_video_parts(
            self, frame_idxs_to_sample: List[List[int]]
    ) -> List[Path]:
        if self.video_metadata.max_cpu_cores_to_pool > 1:
            available_cpus = mp.cpu_count()
            if available_cpus > self.video_metadata.max_cpu_cores_to_pool:
                num_processes = self.video_metadata.max_cpu_cores_to_pool
            else:
                num_processes = available_cpus
            with mp.Pool(num_processes) as p:
                filepaths_to_all_video_parts = p.map(
                    self._write_videoslice_to_disk_for_multiprocessing, enumerate(frame_idxs_to_sample)
                )

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

    def _write_videoslice_to_disk_for_multiprocessing(
            self, idx_and_idxs_of_frames_to_sample: Tuple
    ) -> Path:
        idx, idxs_of_frames_to_sample = idx_and_idxs_of_frames_to_sample
        part_id = str(idx).zfill(3)
        filepath_video_part = self._write_video_to_disk(
            idxs_of_frames_to_sample=idxs_of_frames_to_sample,
            target_fps=self.target_fps,
            part_id=part_id,
        )
        return filepath_video_part

    def _write_video_to_disk(
            self,
            idxs_of_frames_to_sample: Union[int, List[int]],
            target_fps: int,
            part_id: Optional[str] = None,
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
            if part_id is None:
                filepath = self.output_directory.joinpath(
                    f"{self.video_metadata.recording_date}_{self.video_metadata.cam_id}_synchronized_downsampled{self.target_fps}fps.mp4"
                )
            else:
                filepath = self.output_directory.joinpath(
                    f"{self.video_metadata.recording_date}_{self.video_metadata.cam_id}_synchronized_part_{part_id}.mp4"
                )

        else:
            if part_id is None:
                filepath = self.output_directory.joinpath(
                    f"{self.video_metadata.mouse_id}_{self.video_metadata.recording_date}_{self.video_metadata.paradigm}_{self.video_metadata.cam_id}_synchronized_downsampled{self.target_fps}fps.mp4"
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
                video_part_streams[0], filename=str(filepath_concatenated_video)
            )
        output_stream.run(overwrite_output=True, quiet=True)
        return filepath_concatenated_video


class CharucoVideoSynchronizer(Synchronizer):
    def _create_h5_filepath(self, tag: str = "_rawfps_unsynchronized", filtered: bool = False) -> Path:
        pass

    def _adjust_video_to_target_fps_and_run_marker_detection(
            self,
            target_fps: int,
            start_idx: int,
            offset: float,
            test_mode: bool,
            synchronize_only: bool = True,
    ) -> Tuple[None, Path]:
        return None, self._downsample_video(
            start_idx=start_idx,
            offset=offset,
            target_fps=self.target_fps,
            test_mode=test_mode,
        )


class RecordingVideoSynchronizer(Synchronizer):
    def _adjust_video_to_target_fps_and_run_marker_detection(self, target_fps: int, start_idx: int, offset: float,
                                                             test_mode: bool, synchronize_only: bool) -> Tuple[Optional[Path], Optional[Path]]:
        pass

    def _create_h5_filepath(
            self, tag: str = "_rawfps_unsynchronized", filtered: bool = False
    ) -> Path:
        if filtered:
            h5_filepath = self.output_directory.joinpath(
                f"{self.video_metadata.mouse_id}_{self.video_metadata.recording_date}_{self.video_metadata.paradigm}_{self.video_metadata.cam_id}{tag}_filtered.h5"
            )
        else:
            h5_filepath = self.output_directory.joinpath(
                f"{self.video_metadata.mouse_id}_{self.video_metadata.recording_date}_{self.video_metadata.paradigm}_{self.video_metadata.cam_id}{tag}.h5"
            )
        return h5_filepath

    def _run_deep_lab_cut_for_marker_detection(
            self, video_filepath: Path, test_mode: bool = False
    ) -> Path:
        output_filepath = self._create_h5_filepath()

        if (not test_mode) or (test_mode and not output_filepath.exists()):
            config_filepath = self.video_metadata.processing_filepath
            dlc_interface = DeeplabcutInterface(
                object_to_analyse=video_filepath,
                output_directory=self.output_directory,
                marker_detection_directory=config_filepath,
            )
            dlc_ending = dlc_interface.analyze_objects(
                filtering=True, use_gpu=self.use_gpu, filepath=output_filepath
            )

        return output_filepath

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
                manual_interface.analyze_objects(filepath=output_filepath.with_suffix(".h5"))

        return output_filepath


class RecordingVideoDownSynchronizer(RecordingVideoSynchronizer):
    def _adjust_video_to_target_fps_and_run_marker_detection(
            self,
            start_idx: int,
            offset: float,
            target_fps: int = 30,
            test_mode: bool = False,
            synchronize_only: bool = False
    ) -> Tuple[Path, None]:

        downsynchronized_filepath = self._create_h5_filepath(tag=f"_temp")

        if self.video_metadata.processing_type == "DLC":
            full_h5_filepath = self._run_deep_lab_cut_for_marker_detection(
                video_filepath=self.video_metadata.filepath, test_mode=test_mode
            )
            if self.synchro_metadata["use_2D_filter"]:
                full_h5_filepath = self._create_h5_filepath(filtered=True)
        elif self.video_metadata.processing_type == "manual":
            full_h5_filepath = self._run_manual_marker_detection(
                video_filepath=self.video_metadata.filepath, test_mode=test_mode
            )
        else:
            raise ValueError("For processing only DLC and manual are supported!")

        df = pd.read_hdf(full_h5_filepath)
        frame_idxs_to_sample = self._get_sampling_frame_idxs(start_idx=start_idx, offset=offset, target_fps=target_fps)
        new_df = df.loc[frame_idxs_to_sample, :]
        new_df.to_hdf(downsynchronized_filepath, "key")
        return downsynchronized_filepath, None


class RecordingVideoUpSynchronizer(RecordingVideoSynchronizer):
    def _adjust_video_to_target_fps_and_run_marker_detection(
            self,
            target_fps: int,
            start_idx: int,
            offset: float,
            test_mode: bool,
            synchronize_only: bool = False,
    ) -> Tuple[Path, None]:

        upsynchronized_filepath = self._create_h5_filepath(tag=f"_temp")

        if (not test_mode) or (test_mode and not upsynchronized_filepath.exists()):

            if self.video_metadata.processing_type == "DLC":
                full_h5_filepath = self._run_deep_lab_cut_for_marker_detection(
                    video_filepath=self.video_metadata.filepath, test_mode=test_mode
                )
                if self.synchro_metadata["use_2D_filter"]:
                    full_h5_filepath = self._create_h5_filepath(filtered=True)
            elif self.video_metadata.processing_type == "manual":
                full_h5_filepath = self._run_manual_marker_detection(
                    video_filepath=self.video_metadata.filepath, test_mode=test_mode
                )
            else:
                raise ValueError("For processing only DLC and manual are supported!")

            df = pd.read_hdf(full_h5_filepath)
            len_frame_in_ms = 1 / self.video_metadata.fps * 1000
            len_targetframe_in_ms = 1 / self.target_fps * 1000
            total_offset_in_ms = start_idx * len_frame_in_ms + offset
            recording_length = (df.shape[0] - 1) * len_frame_in_ms
            index_in_ms = df.index * len_frame_in_ms
            new_indices = np.arange(
                total_offset_in_ms, recording_length, len_targetframe_in_ms
            )
            d = scipy.interpolate.interp1d(index_in_ms, df, axis=0)
            upsampled = d(new_indices)
            new_df = pd.DataFrame(upsampled, columns=df.columns)
            new_df.to_hdf(str(upsynchronized_filepath), "key")
        return upsynchronized_filepath, None
