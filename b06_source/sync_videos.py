from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union

from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio as iio
import ffmpeg

import cupy as cp

import sys
sys.path.append('/home/ds/GitHub_repos/rapidAligner/')
import rapidAligner as ra


# ToDo: install rapidAligner from GitHub Repo as package to make import more convenient / compatible across systems
# ToDo: function to automatically delete video parts ("delete_individual_parts" default to True)


class TimeseriesTemplate(ABC):
    
    @property
    @abstractmethod
    def template_attribute_string(self) -> str:
        # specifies the attribute name where the template timeseries is saved.
        # will be used by the adjust_template_timeseries_to_fps method
        pass
    
    
    def adjust_template_timeseries_to_fps(self, fps: int) -> List[np.ndarray]:
        template_timeseries = getattr(self, self.template_attribute_string)
        fps_adjusted_templates = []
        framerate = fps/1000
        max_frames = int(template_timeseries.shape[0] * framerate)
        max_offset = 1000 // fps
        for offset_in_ms in range(max_offset):
            image_timestamps = np.linspace(0+offset_in_ms, template_timeseries.shape[0]+offset_in_ms, max_frames, dtype='int')
            while image_timestamps[-1] >= template_timeseries.shape[0]:
                image_timestamps = image_timestamps[:-1]
            adjusted_template = template_timeseries[image_timestamps].copy()
            fps_adjusted_templates.append((adjusted_template, offset_in_ms))
        return fps_adjusted_templates



class MotifTemplate(TimeseriesTemplate):
    
    @property
    def template_attribute_string(self) -> str:
        return 'template_timeseries'
    
    
    def __init__(self, led_on_time_in_ms: int, on_off_period_length_in_ms: int, motif_duration_in_ms: int):
        self.led_on_time_in_ms = led_on_time_in_ms
        self.on_off_period_length_in_ms = on_off_period_length_in_ms
        self.motif_duration_in_ms = motif_duration_in_ms
        self.template_timeseries = self._compute_template_timeseries()
        
    
    def _compute_template_timeseries(self) -> np.ndarray:
        led_on_off_period = np.zeros((self.on_off_period_length_in_ms), dtype='float')
        led_on_off_period[1:self.led_on_time_in_ms+1] = 1
        full_repetitions = self.motif_duration_in_ms // self.on_off_period_length_in_ms
        remaining_ms = self.motif_duration_in_ms % self.on_off_period_length_in_ms
        motif_template = np.concatenate([led_on_off_period]*(full_repetitions+1))
        adjusted_end_index = self.on_off_period_length_in_ms*full_repetitions + remaining_ms
        return motif_template[:adjusted_end_index]


class MultiMotifTemplate(TimeseriesTemplate):
    
    @property
    def template_attribute_string(self) -> str:
        return 'multi_motif_template'
    
    
    def __init__(self) -> None:
        self.motif_templates = []
        
    
    def add_motif_template(self, motif_template: MotifTemplate) -> None:
        self.motif_templates.append(motif_template)
        self.multi_motif_template = self._update_session_template()
    
    
    def _update_session_template(self) -> np.ndarray:
        individual_motif_template_timeseries = [elem.template_timeseries for elem in self.motif_templates]
        return np.concatenate(individual_motif_template_timeseries)



class SingleCamRawCalibrationData:
    
    # ToDo: refactor; probably split into separate classes
    # ToDo: code to extract reliable LED center coords from DLC tracking
    # ToDo: make max_frame_count for splitting video into parts adjustable
    # ToDo: function to join individual .mp4 parts together?
    # ToDo: CPU-based alternative to rapid_Aligner?
    
    def __init__(self, filepath_video: Path, cam_id: str) -> None:
        self.filepath_video = filepath_video
        self.cam_id = cam_id
        self.fps = self._load_fps_from_video_metadata()

        
    def extract_led_pixel_intensities_as_timeseries(self, filepath_tracking: Optional[Path]=None, led_marker_id: Optional[str]=None,
                                                    led_center_row_col_idxs: Optional[Tuple[int, int]]=None, box_rows: int=5, box_cols: int=5) -> None:
        if (filepath_tracking == None) & (led_marker_id == None) & (led_center_row_col_idxs == None):
            raise ValueError('You have to provide either the "filepath_tracking" (i.e. the filepath '
                             'of the DeepLabCut prediction) and the "led_marker_id", OR you can pass '
                             'the x- and y-coordinates of the LEDs center directly as "led_center_row_col_idxs"'
                             ' (for instance if they were determined manually).')
        if led_center_row_col_idxs != None:
            self.led_center_row_col_idxs = led_center_row_col_idxs
        else:    
            self.led_center_row_col_idxs = self._get_led_coords_from_dlc_tracking(filepath = filepath_tracking, marker_id = led_marker_id)
        self.led_timeseries = self._extract_led_pixel_intensities(box_rows = box_rows, box_cols = box_cols)
        
        
    def _load_fps_from_video_metadata(self) -> int:
        fps = iio.v3.immeta(self.filepath_video)['fps']
        if fps % 1 != 0:
            print(f'Warning! The fps of the video is not an integer! -> fps: {fps}')
        return int(fps)
        
    
    def _get_led_coords_from_dlc_tracking(self, filepath: Path, marker_id: str) -> Tuple[int, int]:
        # ToDo:
        # load dlc output file
        # go over all predicted led marker positions
        # calculate median position from those with highest prediction probabilities
        # and confirm that they don´t include large shifts (e.g. very low z-scores only)
        # return format has to be (row_index, column_index)!!!!
        # for now, we will store all manually determined LED center coords here:
        """
        Calibration 11.08:
        coords = {'top': (249, 433),
                  'bottom': (509, 581),
                  'bottom_b': (536, 561),
                  'side1': (299, 560),
                  'side2': (293, 304)}
                  
        # calibration 18.08:
        coords = {'bottom_crop': (503, 670),
                  'bottom_nocrop': (823, 834),
                  'Side1_crop': (456, 1168),
                  'Side1_nocrop': (456, 1168),
                  'Side2_crop': (425, 751),
                  'Side2_nocrop': (425, 751),
                  'top_crop': (336, 537),
                  'top_nocrop': (336, 537)}
        coords = {'bottom': (474, 556),
                  'Side1': (218, 430),
                  'Side2': (227, 331),
                  'Ground1': (429, 378),
                  'Ground2': (300, 418),
                  'top': (270, 500)}
        """
        pass
    
    
    def _extract_led_pixel_intensities(self, box_rows: int, box_cols: int) ->np.ndarray:
        box_row_indices = self._get_start_end_indices_from_center_coord_and_length(center_px = self.led_center_row_col_idxs[0],
                                                                                       length = box_rows)
        box_col_indices = self._get_start_end_indices_from_center_coord_and_length(center_px = self.led_center_row_col_idxs[1],
                                                                                       length = box_cols)
        mean_pixel_intensities = []
        for frame in iio.v3.imiter(self.filepath_video):
            box_mean_intensity = frame[box_row_indices[0]:box_row_indices[1], 
                                       box_col_indices[0]:box_col_indices[1]].mean()
            mean_pixel_intensities.append(box_mean_intensity)
        return np.asarray(mean_pixel_intensities)
                        
                        
    def _get_start_end_indices_from_center_coord_and_length(self, center_px: int, length: int) -> Tuple[int, int]:
        start_index = center_px - (length // 2)
        end_index = center_px + (length - (length // 2))
        return start_index, end_index
    
    
    def find_best_match_of_template(self, template: Union[MotifTemplate, MultiMotifTemplate], start_time: int=0, end_time: int=-1, plot_result: bool=True) -> Tuple[int, int]:
        adjusted_motif_timeseries = template.adjust_template_timeseries_to_fps(fps = self.fps)
        start_frame_idx = self._get_frame_index_clostest_to_time(time = start_time)
        end_frame_idx = self._get_frame_index_clostest_to_time(time = end_time)
        best_match_offset, best_match_start_idx = self._get_offset_and_start_index_of_best_match(adjusted_templates = adjusted_motif_timeseries,
                                                                                                 start_frame_idx = start_frame_idx, 
                                                                                                 end_frame_idx = end_frame_idx)
        if plot_result:
            self._plot_best_alignment_result(template = adjusted_motif_timeseries[best_match_offset][0], start_idx = best_match_start_idx)
        return best_match_offset, best_match_start_idx
    
    
    #ToDo: find_best_match_of_session_template()
    
    
    def _get_frame_index_clostest_to_time(self, time: int) -> int:
        time_error_message = (f'The specified time: {time} is invalid! Both times have to be an integer '
                              'larger than -1, where -1 represents the very last timestamp in the video '
                              'and every other integer (e.g.: 1000) the time in ms. Please be aware, that '
                              '"start_time" has to be larger than "end_time" (with end_time == -1 as only '
                              'exception) and must be smaller or equal to the total video recording time.')
        if time == -1:
            return -1
        else:
            if time < 0:
                raise ValueError(time_error_message)
            framerate = 1000 / self.fps
            closest_frame_idx = round(time / framerate)
            if closest_frame_idx >= self.led_timeseries.shape[0]:
                raise ValueError(time_error_message)
            return closest_frame_idx

        
    def _get_offset_and_start_index_of_best_match(self, adjusted_templates: List[Tuple[np.ndarray, int]], start_frame_idx: int, end_frame_idx: int) -> Tuple[int, int]:
        lowest_sum_of_squared_error_per_template = []
        for template_timeseries, _ in adjusted_templates:
            alignment_results = self._run_rapid_aligner(query = template_timeseries, subject = self.led_timeseries[start_frame_idx:end_frame_idx])
            alignment_results_as_np_array = alignment_results.get()
            lowest_sum_of_squared_error_per_template.append(alignment_results_as_np_array.min())
        lowest_sum_of_squared_error_per_template = np.asarray(lowest_sum_of_squared_error_per_template)
        best_matching_template_index = int(lowest_sum_of_squared_error_per_template.argmin())
        best_alignment_results = self._run_rapid_aligner(query=adjusted_templates[best_matching_template_index][0], subject=self.led_timeseries[start_frame_idx:end_frame_idx])
        start_index = int(best_alignment_results.argmin())
        return best_matching_template_index, start_index


    def _run_rapid_aligner(self, query: np.ndarray, subject: np.ndarray) -> np.ndarray:
        subject_timeseries_gpu = cp.asarray(subject)
        query_timeseries_gpu = cp.asarray(query)
        return ra.ED.zdist(query_timeseries_gpu, subject_timeseries_gpu, mode="fft")
    
    
    def _plot_best_alignment_result(self, template: np.ndarray, start_idx: int) -> None:
        end_idx = start_idx + template.shape[0]
        fig = plt.figure(figsize=(9, 6), facecolor='white')
        gs = fig.add_gridspec(2, 1)
        ax_raw = fig.add_subplot(gs[0,0])
        ax_raw.plot(self.led_timeseries[start_idx:end_idx])
        ax_raw.plot(template)
        ax_zscored = fig.add_subplot(gs[1,0])
        ax_zscored.plot(self._zscore(array = self.led_timeseries[start_idx:end_idx]))
        ax_zscored.plot(self._zscore(array = template))
        plt.show()
        
        
    def _zscore(self, array: np.ndarray) -> np.ndarray:
        return (array-np.mean(array))/np.std(array, ddof=0)
    
    
    def write_synchronized_and_fps_adjusted_calibration_video(self, start_frame_idx: int, offset: int, target_fps: int, max_frame_count: int=3_000) -> None:
        original_ms_per_frame = self._get_ms_interval_per_frame(fps = self.fps)
        led_timeseries_synchronized_to_motif_start = self._crop_led_timeseries_to_motif_start(start_frame_idx = start_frame_idx, offset = offset)
        offset_adjusted_timestamps_synchronized_led_timeseries = self._adjust_timestamps_for_offset(offset = offset, 
                                                                                                    synchronized_timeseries = led_timeseries_synchronized_to_motif_start)
        target_fps_adjusted_timestamps = self._get_timestamps_of_target_fps(target_fps = target_fps, 
                                                                            synchronized_timeseries = led_timeseries_synchronized_to_motif_start)
        idxs_of_frames_to_sample = self._find_frame_idxs_closest_to_target_timestamps(target_timestamps = target_fps_adjusted_timestamps, 
                                                                                      original_timestamps = offset_adjusted_timestamps_synchronized_led_timeseries)
        self.synchronized_and_fps_adjusted_led_timeseries = led_timeseries_synchronized_to_motif_start[idxs_of_frames_to_sample]
        idxs_of_frames_to_sample_adjusted_for_synchronization = self._adjust_frame_idxs_for_synchronization(frame_idxs = idxs_of_frames_to_sample,
                                                                                                            start_frame_idx = start_frame_idx,
                                                                                                            offset = offset)
        frame_idxs_per_part = self._split_into_ram_digestable_parts(idxs_of_frames_to_sample = idxs_of_frames_to_sample_adjusted_for_synchronization, max_frame_count = max_frame_count)
        self._initiate_writing_of_individual_video_parts(frame_idxs_per_part = frame_idxs_per_part, target_fps = 30)
        self._concatenate_individual_video_parts_on_disk()                          
    
    
    def _get_ms_interval_per_frame(self, fps: int) -> float:
        return 1000 / fps
    
        
    def _crop_led_timeseries_to_motif_start(self, start_frame_idx: int, offset: int) -> np.ndarray:
        offset_adjusted_start_frame_idx = start_frame_idx + self._get_n_frames_to_adjust_for_offset(offset = offset)
        return self.led_timeseries[offset_adjusted_start_frame_idx:].copy()
                                                                                                            
        
    def _get_n_frames_to_adjust_for_offset(self, offset: int) -> int:
        return round(offset/self.fps)
    
    
    def _adjust_timestamps_for_offset(self, offset: int, synchronized_timeseries: np.ndarray) -> np.ndarray:
        original_ms_per_frame = self._get_ms_interval_per_frame(fps = self.fps)
        n_frames_to_adjust_for_offset = self._get_n_frames_to_adjust_for_offset(offset = offset)
        updated_offset_after_frame_adjustment = offset - n_frames_to_adjust_for_offset*original_ms_per_frame
        offset_adjusted_timestamps_led_series = np.arange((synchronized_timeseries.shape[0])*original_ms_per_frame, step=original_ms_per_frame)
        return offset_adjusted_timestamps_led_series + updated_offset_after_frame_adjustment
        
    
    def _get_timestamps_of_target_fps(self, target_fps: int, synchronized_timeseries: np.ndarray) -> np.ndarray:
        target_ms_per_frame = self._get_ms_interval_per_frame(fps = target_fps)
        original_ms_per_frame = self._get_ms_interval_per_frame(fps = self.fps)
        max_frames = int((synchronized_timeseries.shape[0] * original_ms_per_frame) / target_ms_per_frame)
        return np.arange(max_frames*target_ms_per_frame, step=target_ms_per_frame)
        
        
    def _find_frame_idxs_closest_to_target_timestamps(self, target_timestamps: np.ndarray, original_timestamps: np.ndarray) -> List[int]:
        frame_indices_closest_to_target_timestamps = []
        for timestamp in target_timestamps:
            closest_frame_index = self._find_closest_timestamp_index(original_timestamps=original_timestamps, timestamp=timestamp)
            frame_indices_closest_to_target_timestamps.append(closest_frame_index)
        return frame_indices_closest_to_target_timestamps
    

    def _find_closest_timestamp_index(self, original_timestamps: np.ndarray, timestamp: float) -> int:
        return np.abs(original_timestamps - timestamp).argmin()
    
    
    def _adjust_frame_idxs_for_synchronization(self, frame_idxs: List[int], start_frame_idx: int, offset: int) -> List[int]:
        offset_adjusted_start_frame_idx = start_frame_idx + self._get_n_frames_to_adjust_for_offset(offset = offset)
        frame_idxs_array = np.asarray(frame_idxs)
        offset_adjusted_frame_idxs = frame_idxs_array + offset_adjusted_start_frame_idx
        return list(offset_adjusted_frame_idxs)
        
        
    def _split_into_ram_digestable_parts(self, idxs_of_frames_to_sample: List[int], max_frame_count: int) -> List[List[int]]:
        frame_idxs_per_part = []
        while len(idxs_of_frames_to_sample) > max_frame_count:
            frame_idxs_per_part.append(idxs_of_frames_to_sample[:max_frame_count])
            idxs_of_frames_to_sample = idxs_of_frames_to_sample[max_frame_count:]
        frame_idxs_per_part.append(idxs_of_frames_to_sample)
        return frame_idxs_per_part
    
    
    def _initiate_writing_of_individual_video_parts(self, frame_idxs_per_part: List[List[int]], target_fps: int) -> None:
        for idx, idxs_of_frames_to_sample in enumerate(frame_idxs_per_part):
            part_id = str(idx).zfill(3)
            self._write_video_to_disk(idxs_of_frames_to_sample = idxs_of_frames_to_sample, target_fps = target_fps, part_id = part_id)
        
        
    def _write_video_to_disk(self, idxs_of_frames_to_sample: List[int], target_fps: int, part_id: Optional[int]=None) -> None:
        selected_frames = []
        print('load original video')
        for i, frame in enumerate(iio.v3.imiter(self.filepath_video)):
            if i > idxs_of_frames_to_sample[-1]:
                break
            if i in idxs_of_frames_to_sample:
                selected_frames.append(frame)
        video_array = np.asarray(selected_frames)
        if part_id == None:
            filepath_out = self.filepath_video.parent.joinpath(f'{self.cam_id}_cam_synchronized_for_calibration.mp4')
        else:
            filepath_out = self.filepath_video.parent.joinpath(f'{self.cam_id}_cam_synchronized_for_calibration_part_{part_id}.mp4')
        print('writing video to disk')
        iio.mimwrite(filepath_out, video_array, fps=target_fps)
        print('done!')


    def _concatenate_individual_video_parts_on_disk(self) -> None:
        io_directory_path = self.filepath_video.parent
        filenames_video_parts = [filename for filename in self._listdir_nohidden(io_directory_path) if filename.startswith(self.cam_id)]
        video_part_streams = [ffmpeg.input(io_directory_path.joinpath(filename)) for filename in filenames_video_parts]
        if len(video_part_streams) >= 2:
            concatenated_video = ffmpeg.concat(video_part_streams[0], video_part_streams[1])
            if len(video_part_streams) >= 3:
                for part_stream in video_part_streams[2:]:
                    concatenated_video = ffmpeg.concat(concatenated_video, part_stream)
        output_stream = ffmpeg.output(concatenated_video, filename=io_directory_path.joinpath(f'{self.cam_id}_cam_synchronized_for_calibration_all_parts.mp4'))
        output_stream.run(overwrite_output = True)

        
    def _listdir_nohidden(self, path: Path) -> List:
        return [f for f in os.listdir(path) if f.startswith('.') == False]