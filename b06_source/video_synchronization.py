from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union

from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio as iio
import ffmpeg

from b06_source.video_metadata import VideoMetadata
from b06_source.utils import Coordinates


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



class Synchronizer(ABC):
    
    def __init__(self, video_metadata: VideoMetadata, use_gpu: bool) -> None:
        self.video_metadata = video_metadata
        self.use_gpu = use_gpu
        
    @property
    @abstractmethod
    def target_fps(self) -> int:
        pass
        
    
    @abstractmethod
    def _adjust_video_to_target_fps_and_run_marker_detection(self, target_fps: int, start_idx: int, offset: float) -> Path:
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

    
    def run_synchronization(self) -> Path:
        led_center_coordinates = self._get_LED_center_coordinates()
        self.led_timeseries = self._extract_led_pixel_intensities_timeseries(led_center_coords = led_center_coordinates)
        template_blinking_motif = self._construct_template_motif(blinking_patterns_metadata = self.video_metadata.configs['blinking_patterns'])
        offset_adjusted_start_idx, remaining_offset= self._find_best_match_of_template(template = template_blinking_motif,
                                                                                                    start_time = 0,
                                                                                                    end_time = 60_000) # ToDo - make start & end time adaptable?
        self.led_timeseries_for_cross_video_validation = self._adjust_led_timeseries_for_cross_validation(start_idx = offset_adjusted_start_idx, offset = remaining_offset)
        return self._adjust_video_to_target_fps_and_run_marker_detection(target_fps = self.target_fps, start_idx = offset_adjusted_start_idx, offset = remaining_offset)


    def _get_LED_center_coordinates(self) -> Coordinates:
        # call marker detector to detect the LED on some 100 frames
        return led_center_coords


    def _extract_led_pixel_intensities(self, led_center_coords: Coordinates, box_rows: int=5, box_cols: int=5) ->np.ndarray:
        box_row_indices = self._get_start_end_indices_from_center_coord_and_length(center_px = led_center_coords.row,
                                                                                   length = box_rows)
        box_col_indices = self._get_start_end_indices_from_center_coord_and_length(center_px = led_center_coords.column,
                                                                                   length = box_cols)
        mean_pixel_intensities = self._calculate_mean_pixel_intensities(box_row_indices = box_row_indices, box_col_indices = box_col_indices)
        return np.asarray(mean_pixel_intensities)   
    
                        
    def _get_start_end_indices_from_center_coord_and_length(self, center_px: int, length: int) -> Tuple[int, int]:
        start_index = center_px - (length // 2)
        end_index = center_px + (length - (length // 2))
        return start_index, end_index    
    

    def _calculate_mean_pixel_intensities(self, box_row_indices: Tuple[int, int], box_col_indices: Tuple[int, int]) -> List[float]:
        mean_pixel_intensities = []
        for frame in iio.v3.imiter(self.filepath):
            box_mean_intensity = frame[box_row_indices[0]:box_row_indices[1], 
                                       box_col_indices[0]:box_col_indices[1]].mean()
            mean_pixel_intensities.append(box_mean_intensity)
        return mean_pixel_intensities

    
    def _find_best_match_of_template(self, template: Union[MotifTemplate, MultiMotifTemplate], start_time: int=0, end_time: int=-1) -> Tuple[int, float]:
        adjusted_motif_timeseries = template.adjust_template_timeseries_to_fps(fps = self.video_metadata.fps)
        start_frame_idx = self._get_frame_index_closest_to_time(time = start_time)
        end_frame_idx = self._get_frame_index_closest_to_time(time = end_time)
        best_match_start_frame_idx, best_match_offset = self._get_start_index_and_offset_of_best_match(adjusted_templates = adjusted_motif_timeseries,
                                                                                                       start_frame_idx = start_frame_idx, 
                                                                                                       end_frame_idx = end_frame_idx)
        offset_adjusted_start_idx, remaining_offset = self._adjust_start_index_and_offset(start_frame_idx = best_match_start_frame_idx,
                                                                                                           offset = best_match_offset,
                                                                                                           fps = self.video_metatdata.fps)
        return offset_adjusted_start_idx, remaining_offset


    def _get_frame_index_closest_to_time(self, time: int) -> int:
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
            framerate = 1000 / self.fps # adapt
            closest_frame_idx = round(time / framerate)
            if closest_frame_idx >= self.led_timeseries.shape[0]:
                raise ValueError(time_error_message)
            return closest_frame_idx


    def _get_start_index_and_offset_of_best_match(self, adjusted_templates: List[Tuple[np.ndarray, int]], start_frame_idx: int, end_frame_idx: int) -> Tuple[int, int]:
        # ToDo - check if error is within range to automatically accept as good match, otherwise report file & save plots for inspection
        lowest_sum_of_squared_error_per_template = []
        for template_timeseries, _ in adjusted_templates:
            alignment_results = self._run_alignment(query = template_timeseries, subject = self.led_timeseries[start_frame_idx:end_frame_idx])
            lowest_sum_of_squared_error_per_template.append(alignment_results.min())
        lowest_sum_of_squared_error_per_template = np.asarray(lowest_sum_of_squared_error_per_template)
        best_matching_template_index = int(lowest_sum_of_squared_error_per_template.argmin())
        best_alignment_results = self._run_alignment(query=adjusted_templates[best_matching_template_index][0], subject=self.led_timeseries[start_frame_idx:end_frame_idx])
        start_index = int(best_alignment_results.argmin())
        return start_index, best_matching_template_index


    def _adjust_start_idx_and_offset(self, start_frame_idx: int, offset: int, fps: int) -> Tuple[int, float]:
        n_frames_to_add = round(offset / fps)
        adjusted_start_frame_idx = start_frame_idx + n_frames_to_add
        original_ms_per_frame = self._get_ms_interval_per_frame(fps = fps)
        remaining_offset = offset - n_frames_to_add*original_ms_per_frame
        return adjusted_start_frame_idx, remaining_offset


    def _run_alignment(self, query: np.ndarray, subject: np.ndarray) -> np.ndarray:
        if self.use_gpu:
            alignment_results = self._run_rapid_aligner(query = query, subject = subject)
        else:
            alignment_results = self._run_cpu_aligner(query = query, subject = subject)
        return alignment_results
    
    
    def _run_rapid_aligner(self, query: np.ndarray, subject: np.ndarray) -> np.ndarray:
        import sys
        sys.path.append('/home/ds/GitHub_repos/rapidAligner/')
        import cupy as cp
        import rapidAligner as ra
        subject_timeseries_gpu = cp.asarray(subject)
        query_timeseries_gpu = cp.asarray(query)
        alignment_results = ra.ED.zdist(query_timeseries_gpu, subject_timeseries_gpu, mode="fft")
        return alignment_results.get()
    
    
    def _run_cpu_aligner(self, query: np.ndarray, subject: np.ndarray) -> np.ndarray:
        # same as rapidAligner, just using numpy instead of cupy
        return self._fft_zdist(q = query, s = subject, epsilon=1e-6)

    
    def _fft_zdist(self, q: np.ndarray, s: np.ndarray, epsilon: float):
        alignment, kahan = 10_000, 0
        m, q = len(q), self._znorm(q, epsilon)
        n = (len(s)+alignment-1)//alignment*alignment
        is_ = np.zeros(n, dtype=s.dtype)
        is_[:len(s)] = s
        delta = n-len(s)
        x, y = self._cumsum(is_, kahan), self._cumsum(is_**2, kahan)
        x = x[+m:]-x[:-m]
        y = y[+m:]-y[:-m]
        z = np.sqrt(np.maximum(y/m-np.square(x/m), 0))
        e = np.zeros(n, dtype=q.dtype)
        e[:m] = q
        r = np.fft.irfft(np.fft.rfft(e).conj()*np.fft.rfft(is_), n=n)
        f = np.where(z > 0 , 2*(m-r[:-m+1]/z), m*np.ones_like(z))
        return f[:len(s)-m+1]

    
    def _cumsum(self, x, kahan=0):
        assert(isinstance(kahan, int) and kahan >= 0)
        y = np.empty(len(x)+1, dtype=x.dtype)
        y[0] = 0
        np.cumsum(x, out=y[1:])
        if kahan:
            r = x-np.diff(y)
            if(np.max(np.abs(r))):
                y += cumsum(r, kahan-1)
        return y


    def _znorm(self, x, epsilon):
        return (x-np.mean(x))/max(np.std(x, ddof=0), epsilon)        
    

    def _plot_best_alignment_result(self, template: np.ndarray) -> None:
        end_idx = self.best_match_start_idx + template.shape[0]
        fig = plt.figure(figsize=(9, 6), facecolor='white')
        gs = fig.add_gridspec(2, 1)
        ax_raw = fig.add_subplot(gs[0,0])
        ax_raw.plot(self.led_timeseries[self.best_match_start_idx:end_idx])
        ax_raw.plot(template)
        ax_zscored = fig.add_subplot(gs[1,0])
        ax_zscored.plot(self._zscore(array = self.led_timeseries[self.best_match_start_idx:end_idx]))
        ax_zscored.plot(self._zscore(array = template))
        plt.show()
    
    
    def _zscore(self, array: np.ndarray) -> np.ndarray:
        return (array-np.mean(array))/np.std(array, ddof=0)    


    def _adjust_led_timeseries_for_cross_validation(self, start_idx: int, offset: float) -> np.ndarray:
        adjusted_led_timeseries = self.led_timeseries[start_idx:].copy()
        if self.video_metadata.fps != 30:
            adjusted_led_timeseries = self._downsample_led_timeseries(timeseries = adjusted_led_timeseries, offset = offset)
        return adjusted_led_timeseries
        

    def _downsample_led_timeseries(self, timeseries: np.ndarray, offset: float) -> np.ndarray:
        n_frames_after_downsampling = self._compute_fps_adjusted_frame_count(original_n_frames = timeseries.shape[0], original_fps = self.video_metadata.fps, target_fps = 30)
        original_timestamps = self._compute_timestamps(n_frames = timeseries.shape[0], fps = self.video_metadata.fps, offset = offset)
        target_timestamps = self._compute_timestamps(n_frames = n_frames_after_downsampling, fps = 30)
        frame_idxs_best_matching_timestamps = self._find_frame_idxs_closest_to_target_timestamps(target_timestamps = target_timestamps, original_timestamps = original_timestamps)
        return timeseries[frame_idxs_with_best_matching_timestamps]
        
    
    def _compute_fps_adjusted_frame_count(self, original_n_frames: int, original_fps: int, target_fps: int) -> int:
        target_ms_per_frame = self._get_ms_interval_per_frame(fps = target_fps)
        original_ms_per_frame = self._get_ms_interval_per_frame(fps = original_fps)
        return int((original_n_frames * original_ms_per_frame) / target_ms_per_frame)

    
    def _compute_timestamps(self, n_frames: int, fps: int, offset: Optional[float]=0.0) -> np.ndarray:
        ms_per_frame = self._get_ms_interval_per_frame(fps = fps)
        timestamps = np.arange(n_frames*ms_per_frame, step=ms_per_frame)
        return timestamps + offset

    
    def _find_closest_timestamp_index(self, original_timestamps: np.ndarray, timestamp: float) -> int:
        return np.abs(original_timestamps - timestamp).argmin()        
        
        
    def _find_frame_idxs_closest_to_target_timestamps(self, target_timestamps: np.ndarray, original_timestamps: np.ndarray) -> List[int]:
        frame_indices_closest_to_target_timestamps = []
        for timestamp in target_timestamps:
            closest_frame_index = self._find_closest_timestamp_index(original_timestamps = original_timestamps, timestamp = timestamp)
            frame_indices_closest_to_target_timestamps.append(closest_frame_index)
        return frame_indices_closest_to_target_timestamps    


    def _get_ms_interval_per_frame(self, fps: int) -> float:
        return 1000 / fps


    def _downsample_video(self, start_idx: int, offset: float, target_fps: int=30) -> Path:
        frame_idxs_to_sample = self._get_sampling_frame_idxs(start_idx = start_idx, offset = offset, target_fps = target_fps)
        sampling_frame_idxs_per_part = self._split_into_ram_digestable_parts(idxs_of_frames_to_sample = frame_idxs_to_sample, max_frame_count = 3_000)
        if len(frame_idxs_per_part) > 1:
            filepaths_all_video_parts = self._initiate_iterative_writing_of_individual_video_parts(frame_idxs_per_part = sampling_frame_idxs_per_part, target_fps = target_fps)
            filepath_downsampled_video = self._concatenate_individual_video_parts_on_disk(filepaths_of_video_parts = filepaths_all_video_parts)
            self._delete_individual_video_parts(filepaths_of_video_parts = filepaths_all_video_parts)
        else:
            filepath_downsampled_video = self._write_video_to_disk(idxs_of_frames_to_sample = frame_idxs_per_part[0], target_fps = target_fps)
        return filepath_downsampled_video
        
        
    def _get_sampling_frame_idxs(self, start_idx: int, offset: float, target_fps: int) -> List[int]:
        original_n_frames = self.led_timeseries[start_idx:].shape[0]
        n_frames_after_downsampling = self._compute_fps_adjusted_frame_count(original_n_frames = original_n_frames,
                                                                             original_fps = self.video_metadata.fps,
                                                                             target_fps = target_fps)
        original_timestamps = self._compute_timestamps(n_frames = original_n_frames, fps = self.video_metadata.fps, offset = offset)
        target_timestamps = self._compute_timestamps(n_frames = n_frames_after_downsampling, fps = target_fps)
        frame_idxs_best_matching_timestamps = self._find_frame_idxs_closest_to_target_timestamps(target_timestamps = target_timestamps, original_timestamps = original_timestamps)
        sampling_frame_idxs = self._adjust_frame_idxs_for_synchronization_shift(unadjusted_frame_idxs = frame_idxs_best_matching_timestamps, start_idx = start_idx)
        return sampling_frame_idxs
        
        
    def _adjust_frame_idxs_for_synchronization_shift(self, unadjusted_frame_idxs: List[int], start_idx: int) -> List[int]:
        adjusted_frame_idxs = np.asarray(unadjusted_frame_idxs) + start_idx
        return list(adjusted_frame_idxs)


    def _split_into_ram_digestable_parts(self, idxs_of_frames_to_sample: List[int], max_frame_count: int) -> List[List[int]]:
        frame_idxs_per_part = []
        while len(idxs_of_frames_to_sample) > max_frame_count:
            frame_idxs_per_part.append(idxs_of_frames_to_sample[:max_frame_count])
            idxs_of_frames_to_sample = idxs_of_frames_to_sample[max_frame_count:]
        frame_idxs_per_part.append(idxs_of_frames_to_sample)
        return frame_idxs_per_part


    def _initiate_iterative_writing_of_individual_video_parts(self, frame_idxs_per_part: List[List[int]], target_fps: int) -> List[Path]:
        filepaths_to_all_video_parts = []
        for idx, idxs_of_frames_to_sample in enumerate(frame_idxs_per_part):
            part_id = str(idx).zfill(3)
            filepath_video_part = self._write_video_to_disk(idxs_of_frames_to_sample = idxs_of_frames_to_sample, target_fps = target_fps, part_id = part_id)
            filepaths_to_all_video_parts.append(filepath_video_part)
        return filepaths_to_all_video_parts

    def _write_video_to_disk(self, idxs_of_frames_to_sample: List[int], target_fps: int, part_id: Optional[int]=None) -> Path:
        selected_frames = []
        print('load original video')
        for i, frame in enumerate(iio.v3.imiter(self.video_metadata.filepath)):
            if i > idxs_of_frames_to_sample[-1]:
                break
            if i in idxs_of_frames_to_sample:
                selected_frames.append(frame)
        video_array = np.asarray(selected_frames)
        print('writing video to disk')
        filepath_out = self._construct_video_filepath(part_id = part_id)
        iio.mimwrite(filepath_out, video_array, fps=target_fps)
        print('done!')
        return filepath_out


    def _construct_video_filepath(self, part_id: Optional[int]) -> Path:
        # ToDo: proper file & directory structure
        # ToDo: include mouse id & session id - OR - charuco
        if part_id == None:
            filepath = self.video_metadata.filepath.parent.joinpath(f'{self.video_metadata.date}_{self.video_metadata.cam_id}_synchronized.mp4')
        else:
            filepath = self.video_metadata.filepath.parent.joinpath(f'{self.video_metadata.date}_{self.video_metadata.cam_id}_synchronized_part_{part_id}.mp4')
        return filepath
    
    
    def _concatenate_individual_video_parts_on_disk(self, filepaths_of_video_parts: List[Path]) -> Path:
        video_part_streams = [ffmpeg.input(filepath) for filepath in filepaths_of_video_parts]
        if len(video_part_streams) >= 2:
            concatenated_video = ffmpeg.concat(video_part_streams[0], video_part_streams[1])
            if len(video_part_streams) >= 3:
                for part_stream in video_part_streams[2:]:
                    concatenated_video = ffmpeg.concat(concatenated_video, part_stream)
        filepath_concatenated_video = self._construct_video_filepath(part_id = None)
        output_stream = ffmpeg.output(concatenated_video, filename=filepath_concatenated_video)
        output_stream.run(overwrite_output = True)
        return filepath_concatenated_video


    def _delete_individual_video_parts(self, filepaths_of_video_parts: List[Path]) -> None:
        for filepath in filepaths_of_video_parts:
            filepath.unlink()




class CharucoVideoSynchronizer(Synchronizer):
    
    @property
    def target_fps(self) -> int:
        return 30
    

    def _adjust_video_to_target_fps_and_run_marker_detection(self, target_fps: int, start_idx: int, offset: float) -> Path:
        return self._downsample_video(start_idx = start_idx, offset = offset, target_fps = target_fps)



class RecordingVideoSynchronizer(Synchronizer):

    def _run_deep_lab_cut_for_marker_detection(self, video_filepath: Path) -> Path:
        # initiate marker detection on filepath
        # save .h5 file
        # return filepath to .h5 file
        pass



class RecordingVideoDownSynchronizer(RecordingVideoSynchronizer):

    @property
    def target_fps(self) -> int:
        return 30
    

    def _adjust_video_to_target_fps_and_run_marker_detection(self, target_fps: int, start_idx: int, offset: float) -> Path:
        downsampled_video_filepath = self._downsample_video(start_idx = start_idx, offset = offset, target_fps = target_fps)
        detected_markers_filepath = self._run_deep_lab_cut_for_marker_detection(video_filepath = downsampled_video_filepath)
        return detected_markers_filepath