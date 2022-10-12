from typing import List, Tuple, Dict, Union, Optional
from abc import ABC, abstractmethod

from pathlib import Path
import pandas as pd
import numpy as np
import math
import pickle
import itertools as it
import imageio as iio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

import aniposelib as ap_lib

from .utils import load_single_frame_of_video

# ToDo:
# Should the SingleCamDataForAnipose & the CalibrationForAnipose3DTracking only be 
#   subclasses of a more general parent that could then also be used as base class
#   to process the actual experimental recordings? Here only triangulation would be
#   needed & calibration should be loaded and error estimation based on test position
#   markers would not be neccessary!

# ToDo:
# Ensure proper filenaming. For instance, it should include the date and the most relevant
# settings. Alternatively, we could create a configs file that is saved in addition, holding
# all important information about the calibration process.


# ToDo:
# Mark the corresponding functions that were copied & then adapted from the anipose Repo!


class IntrinsicCameraCalibrator(ABC):
    #https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
    
    @abstractmethod 
    def _run_camera_type_specific_calibration(self, objpoints: List[np.ndarray], imgpoints: List[np.ndarray]) -> Tuple:
        # wrapper to camera type specific calibration function
        # all remaining data is stored in attributes of the object
        pass


    def __init__(self, filepath_calibration_video: Path, max_frame_count: int) -> None:
        self.video_filepath = filepath_calibration_video
        self.max_frame_count = max_frame_count
        self.video_reader = iio.get_reader(filepath_calibration_video)


    @property
    def checkerboard_rows_and_columns(self) -> Tuple[int, int]:
        return (5, 5)

    @property
    def d(self) -> np.ndarray:
        return np.zeros((4, 1))

    @property
    def imsize(self) -> Tuple[int, int]:
        # ToDo:
        # Check whether it has to be the shape of the grayscale image and what it actually looks like
        frame = np.asarray(self.video_reader.get_data(0))
        frame_in_gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame_in_gray_scale.shape[::-1]

    @property
    def k(self) -> np.ndarray:
        return np.zeros((3, 3))

    @property
    def objp(self) -> np.ndarray:
        objp = np.zeros((1, self.checkerboard_rows_and_columns[0]*self.checkerboard_rows_and_columns[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:self.checkerboard_rows_and_columns[0], 0:self.checkerboard_rows_and_columns[1]].T.reshape(-1, 2)
        return objp

    @property
    def subpixel_criteria(self) -> Tuple:
        return (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)


    def run(self) -> Dict:
        selected_frame_idxs = self._get_indices_of_selected_frames()
        detected_checkerboard_corners_per_image = self._detect_checkerboard_corners(frame_idxs = selected_frame_idxs)
        if len(detected_checkerboard_corners_per_image) != self.max_frame_count:
            detected_checkerboard_corners_per_image = self._attempt_to_match_max_frame_count(corners_per_image = detected_checkerboard_corners_per_image,
                                                                                             already_selected_frame_idxs = selected_frame_idxs)
        object_points = self._compute_object_points(n_detected_boards = len(detected_checkerboard_corners_per_image))
        retval, K, D, rvec, tvec = self._run_camera_type_specific_calibration(objpoints = object_points, imgpoints = detected_checkerboard_corners_per_image)
        calibration_results = self._construct_calibration_results(K = K, D = D, rvec = rvec, tvec = tvec)
        return calibration_results


    def save(self) -> None:
        video_filename = self.video_filepath.name
        filename = f'{video_filename[:video_filename.rfind(".")]}_intrinsic_calibration_results.p'
        with open(self.video_filepath.parent.joinpath(filename), 'wb') as io:
            pickle.dump(self.calibration_results, io)


    def _attempt_to_match_max_frame_count(self, corners_per_image: List[np.ndarray], already_selected_frame_idxs: List[int]) -> List[np.ndarray]:
        print(f'Frames with detected checkerboard: {len(corners_per_image)}.')
        if len(corners_per_image) < self.max_frame_count:
            print('Trying to find some more ...')
            corners_per_image = self._attempt_to_reach_max_frame_count(corners_per_image = corners_per_image, 
                                                                       already_selected_frame_idxs = already_selected_frame_idxs)
            print(f'Done. Now we are at a total of {len(corners_per_image)} frames in which I could detect a checkerboard.')
        elif len(corners_per_image) > self.max_frame_count:
            corners_per_image = self._limit_to_max_frame_count(all_detected_corners = corners_per_image)
            print(f'Limited them to only {len(corners_per_image)}.')
        return corners_per_image


    def _attempt_to_reach_max_frame_count(self, corners_per_image: List[np.ndarray], already_selected_frame_idxs: List[int]) -> List[np.ndarray]:
        # ToDo
        # limit time?
        total_frame_count = self.video_reader.count_frames()
        for idx in range(total_frame_count):
            if len(corners_per_image) < self.max_frame_count:
                if idx not in already_selected_frame_idxs:
                    checkerboard_detected, predicted_corners = self._run_checkerboard_corner_detection(idx = idx)
                    if checkerboard_detected:
                        corners_per_image.append(predicted_corners)
            else:
                break
        return corners_per_image


    def _construct_calibration_results(self, K: np.ndarray, D: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> Dict:
        # ToDo: 
        # - confirm type hints
        # - Potentially add more parameters that might be required for adjusting this intrinsic
        #    calibration to cropping or flipping of the actual recordings (both calibration and experiment)
        # - rvec and tvec probably not required, as they belong to the "extrinsic" parameters, which we
        #    will anyhow compute for the calibrations in anipose
        calibration_results = {"K": K, "D": D, "rvec": rvec, "tvec": tvec, "size": self.imsize}
        setattr(self, 'calibration_results', calibration_results)
        return calibration_results


    def _compute_object_points(self, n_detected_boards: int) -> List[np.ndarray]:
        object_points = []
        for i in range(n_detected_boards):
            object_points.append(self.objp)
        return object_points


    def _detect_checkerboard_corners(self, frame_idxs: List[int]) -> List[np.ndarray]:
        detected_checkerboard_corners_per_image = []
        for idx in frame_idxs:
            checkerboard_detected, predicted_corners = self._run_checkerboard_corner_detection(idx = idx)
            if checkerboard_detected:
                detected_checkerboard_corners_per_image.append(predicted_corners)
        return detected_checkerboard_corners_per_image


    def _determine_sampling_rate(self) -> int:
        total_frame_count = self.video_reader.count_frames()
        if total_frame_count >= 5*self.max_frame_count:
            sampling_rate = total_frame_count // (5*self.max_frame_count)
        else:
            sampling_rate = 1
        return sampling_rate


    def _get_indices_of_selected_frames(self) -> List[int]:
        sampling_rate = self._determine_sampling_rate()
        frame_idxs = self._get_sampling_frame_indices(sampling_rate = sampling_rate)
        return list(frame_idxs)


    def _get_sampling_frame_indices(self, sampling_rate: int) -> np.ndarray:
        total_frame_count = self.video_reader.count_frames()
        n_frames_to_select = total_frame_count // sampling_rate
        return np.linspace(0, total_frame_count, n_frames_to_select, endpoint=False, dtype=int)


    def _limit_to_max_frame_count(self, all_detected_corners: List[np.ndarray]) -> List[np.ndarray]:
        sampling_idxs = np.linspace(0, len(all_detected_corners), self.max_frame_count, endpoint=False, dtype=int)
        sampled_corners = np.asarray(all_detected_corners)[sampling_idxs]
        return list(sampled_corners)


    def _run_checkerboard_corner_detection(self, idx: int) -> Tuple[bool, np.ndarray]:
        image = np.asarray(self.video_reader.get_data(idx))
        gray_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        checkerboard_detected, predicted_corners = cv2.findChessboardCorners(gray_scale_image, 
                                                                             self.checkerboard_rows_and_columns, 
                                                                             cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE)
        #if checkerboard_detected:
            #predicted_corners = cv2.cornerSubPix(gray_scale_image, predicted_corners, (3,3), (-1,-1), self.subpixel_criteria)
        return checkerboard_detected, predicted_corners

        
        

class IntrinsicCalibratorFisheyeCamera(IntrinsicCameraCalibrator):
          
    def _compute_rvecs_and_tvecs(self, n_detected_boards: int) -> Tuple[np.ndarray, np.ndarray]:
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(n_detected_boards)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(n_detected_boards)]
        return rvecs, tvecs


    def _run_camera_type_specific_calibration(self, objpoints: List[np.ndarray], imgpoints: List[np.ndarray]) -> Tuple:
        rvecs, tvecs = self._compute_rvecs_and_tvecs(n_detected_boards = len(objpoints))
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
        new_subpixel_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        return cv2.fisheye.calibrate(objpoints, imgpoints, self.imsize, self.k, self.d, rvecs, tvecs, calibration_flags, new_subpixel_criteria)      
    
    
        
class IntrinsicCalibratorRegularCamera(IntrinsicCameraCalibrator):
    
    def _run_camera_type_specific_calibration(self, objpoints: List[np.ndarray], imgpoints: List[np.ndarray]) -> Tuple:
        return cv2.calibrateCamera(objpoints, imgpoints, self.imsize, None, None)        





class TestPositionsGroundTruth:

    # ToDo:
    # Add method that allows to remove marker id?
    
    # ToDo:
    # include "add_marker_ids_to_be_connected_in_3d_plots" and "reference_distance_ids_with_corresponding_marker_ids" in save & load functions?
    
    def __init__(self) -> None:
        self.marker_ids_with_distances = {}
        self.unique_marker_ids = []
        self.reference_distance_ids_with_corresponding_marker_ids = []
        self.marker_ids_to_connect_in_3D_plot = []
        self._add_maze_corners()
        self.add_marker_ids_to_be_connected_in_3d_plots(marker_ids = ('maze_corner_open_left',
                                                                      'maze_corner_open_right',
                                                                      'maze_corner_closed_right',
                                                                      'maze_corner_closed_left'))
        self.add_marker_ids_and_distance_id_as_reference_distance(marker_ids = ('maze_corner_open_left', 'maze_corner_closed_left'),
                                                                  distance_id = 'maze_length_left')
        self.add_marker_ids_and_distance_id_as_reference_distance(marker_ids = ('maze_corner_open_right', 'maze_corner_closed_right'),
                                                                  distance_id = 'maze_length_right')
    
    def add_new_marker_id(self, marker_id: str, other_marker_ids_with_distances: List[Tuple[str, Union[int, float]]]) -> None:
        for other_marker_id, distance in other_marker_ids_with_distances:
            self._add_ground_truth_information(marker_id_a = marker_id, marker_id_b = other_marker_id, distance = distance)
            self._add_ground_truth_information(marker_id_a = other_marker_id, marker_id_b = marker_id, distance = distance)
            
    
    def add_marker_ids_and_distance_id_as_reference_distance(self, marker_ids: Tuple[str, str], distance_id: str) -> None:
        self.reference_distance_ids_with_corresponding_marker_ids.append((distance_id, marker_ids)) 
    
    
    def add_marker_ids_to_be_connected_in_3d_plots(self, marker_ids: Tuple[str]) -> None:
        if marker_ids not in self.marker_ids_to_connect_in_3D_plot:
            self.marker_ids_to_connect_in_3D_plot.append(marker_ids)


    def load_from_disk(self, filepath: Path) -> None:
        with open(filepath, 'rb') as io:
            marker_ids_with_distances = pickle.load(io)
        unique_marker_ids = list(marker_ids_with_distances.keys())
        setattr(self, 'marker_ids_with_distances', marker_ids_with_distances)
        setattr(self, 'unique_marker_ids', unique_marker_ids)


    def save_to_disk(self, filepath: Path) -> None:
        # ToDo: validate filepath, -name, -extension & provide default alternative
        with open(filepath, 'wb') as io:
            pickle.dump(self.marker_ids_with_distances, io)   

    
    def _add_ground_truth_information(self, marker_id_a: str, marker_id_b: str, distance: Union[int, float]) -> None:
        if marker_id_a not in self.marker_ids_with_distances.keys():
            self.marker_ids_with_distances[marker_id_a] = {}
            self.unique_marker_ids.append(marker_id_a)
        self.marker_ids_with_distances[marker_id_a][marker_id_b] = distance
        
    
    def _add_maze_corners(self) -> None:
        maze_width, maze_length = 4, 50
        maze_diagonal = (maze_width**2 + maze_length**2)**0.5
        maze_corner_distances = {'maze_corner_open_left': [('maze_corner_open_right', maze_width),
                                                           ('maze_corner_closed_right', maze_diagonal),
                                                           ('maze_corner_closed_left', maze_length)],
                                 'maze_corner_open_right': [('maze_corner_closed_right', maze_length),
                                                            ('maze_corner_closed_left', maze_diagonal)],
                                 'maze_corner_closed_left': [('maze_corner_closed_right', maze_width)]}
        for marker_id, distances in maze_corner_distances.items():
            self.add_new_marker_id(marker_id = marker_id, other_marker_ids_with_distances = distances)



class SingleCamDataForAnipose:
    
    # ToDo:
    # We might need some methods to inspect the quality of the intrinsic calibration
    # after adjusting it to the cropped video, since flipping the video streams in
    # ICcapture has different effects, depending on whether it was applied right from
    # launching the software / cameras, or whether it was manually activated once the
    # software is running and the camera was loaded. This is at least our best guess
    # for the inconsistent behavior & warrants further testing.
    
    def __init__(self, cam_id: str, filepath_synchronized_calibration_video: Path, fisheye: bool=False) -> None:
        self.cam_id = cam_id
        self.fisheye = fisheye
        self.filepath_synchronized_calibration_video = filepath_synchronized_calibration_video


    def add_cropping_offsets(self, x_or_column_offset: int=0, y_or_row_offset: int=0) -> None:
        setattr(self, 'cropping_offsets', (x_or_column_offset, y_or_row_offset))
    
    
    def add_flipping_details(self, flipped_horizontally: bool=False, flipped_vertically: bool=False) -> None:
        setattr(self, 'flipped_horizontally', flipped_horizontally)
        setattr(self, 'flipped_vertically', flipped_vertically)
        
    
    def add_rotation_details(self, degrees_rotated_clockwise: int=0) -> None:
        setattr(self, 'degrees_rotated_clockwise', degrees_rotated_clockwise)

        
    def add_manual_test_position_marker(self, marker_id: str, x_or_column_idx: int, y_or_row_idx: int, likelihood: float, overwrite: bool=False) -> None:
        if hasattr(self, 'manual_test_position_marker_coords_pred') == False:
            self.manual_test_position_marker_coords_pred = {}
        if (marker_id in self.manual_test_position_marker_coords_pred.keys()) & (overwrite == False):
            raise ValueError('There are already coordinates for the marker you '
                             f'tried to add: "{marker_id}: {self.manual_test_position_marker_coords_pred[marker_id]}'
                             '". If you would like to overwrite these coordinates, please pass '
                             '"overwrite = True" as additional argument to this method!')
        self.manual_test_position_marker_coords_pred[marker_id] = {'x': [x_or_column_idx], 'y': [y_or_row_idx], 'likelihood': [likelihood]}


    def export_as_aniposelib_Camera_object(self, fisheye: bool) -> ap_lib.cameras.Camera:
        if fisheye:
            camera = ap_lib.cameras.FisheyeCamera(name = self.cam_id,
                                           matrix = self.intrinsic_calibration_for_anipose['K'],
                                           dist = self.intrinsic_calibration_for_anipose['D'],
                                           extra_dist = False)
        else:                                   
            camera = ap_lib.cameras.Camera(name = self.cam_id,
                                   matrix = self.intrinsic_calibration_for_anipose['K'],
                                   dist = self.intrinsic_calibration_for_anipose['D'],
                                   extra_dist = False)
        return camera
    
    
    def inspect_intrinsic_calibration(self, frame_idx: int=0) -> None:
        distorted_input_image = load_single_frame_of_video(filepath = self.filepath_synchronized_calibration_video, frame_idx = frame_idx)
        if self.fisheye:
            undistorted_output_image = self._undistort_fisheye_image_for_inspection(image = distorted_input_image)
        else:
            undistorted_output_image = cv2.undistort(distorted_input_image, 
                                                     self.intrinsic_calibration_for_anipose['K'], 
                                                     self.intrinsic_calibration_for_anipose['D'])
        self._plot_distorted_and_undistorted_image(distorted_image = distorted_input_image, undistorted_image = undistorted_output_image)          


    def load_intrinsic_camera_calibration(self, filepath_intrinsic_calibration: Path) -> None:
        with open(filepath_intrinsic_calibration, 'rb') as io:
            intrinsic_calibration = pickle.load(io)
        adjusting_required = self._is_adjusting_of_intrinsic_calibration_required(unadjusted_intrinsic_calibration = intrinsic_calibration)
        self._set_intrinsic_calibration(intrinsic_calibration = intrinsic_calibration, adjusting_required = adjusting_required)
        
    
    def load_test_position_markers_df_from_dlc_prediction(self, filepath_deeplabcut_prediction: Path) -> None:
        df = pd.read_hdf(filepath_deeplabcut_prediction)
        setattr(self, 'test_position_markers_df', df)
        setattr(self, 'filepath_test_position_marker_prediction', filepath_deeplabcut_prediction)
 
    
    def run_intrinsic_camera_calibration(self, filepath_checkerboard_video: Path, save: bool=True, max_frame_count: int=300) -> None:
        if self.fisheye:
            calibrator = IntrinsicCalibratorFisheyeCamera(filepath_calibration_video = filepath_checkerboard_video, max_frame_count = max_frame_count)
        else:
            calibrator = IntrinsicCalibratorRegularCamera(filepath_calibration_video = filepath_checkerboard_video, max_frame_count = max_frame_count)
        intrinsic_calibration = calibrator.run()
        if save:
            calibrator.save()
        adjusting_required = self._is_adjusting_of_intrinsic_calibration_required(unadjusted_intrinsic_calibration = intrinsic_calibration)
        self._set_intrinsic_calibration(intrinsic_calibration = intrinsic_calibration, adjusting_required = adjusting_required)
        
        
    def validate_and_save_manual_marker_coords_as_fake_dlc_output(self, test_positions_gt: TestPositionsGroundTruth, output_filepath: Optional[Path]=None,
                                                                  add_missing_marker_ids_with_0_likelihood: bool=True) -> None:
        if type(output_filepath) != Path:
            output_filepath = self.filepath_synchronized_calibration_video.parent
        if output_filepath.name.endswith('.h5') == False:
            if output_filepath.is_dir():
                output_filepath = output_filepath.joinpath(f'{self.cam_id}_manual_test_position_marker_fake.h5')
            else:
                output_filepath = output_filepath.parent.joinpath(f'{self.cam_id}_manual_test_position_marker_fake.h5')
        self.test_position_markers_df = self._construct_dlc_output_style_df_from_manual_marker_coords()
        self.validate_test_position_marker_ids(test_positions_gt = test_positions_gt, add_missing_marker_ids_with_0_likelihood = add_missing_marker_ids_with_0_likelihood)
        self.test_position_markers_df.to_hdf(output_filepath, "df")
        print(f'Your dataframe was successfully saved at: {output_filepath.as_posix()}.')
        self.load_test_position_markers_df_from_dlc_prediction(filepath_deeplabcut_prediction = output_filepath)
        print('Your "fake DLC marker perdictions" were successfully loaded to the SingleCamDataForAnipose object!')
            
            
    def save_manual_marker_coords_as_fake_dlc_output(self, output_filepath: Optional[Path]=None):
        # ToDo: this could very well be suitable to become extracted to a utils function
        #       it could then easily be re-used as "validate_output_filename_and_path"
        #       if for instance the extension string, the defaults, and the warning message
        #       can be adapted / passed as arguments!
        if type(output_filepath) != Path:
            output_filepath = self.filepath_synchronized_calibration_video.parent
        if output_filepath.name.endswith('.h5') == False:
            if output_filepath.is_dir():
                output_filepath = output_filepath.joinpath(f'{self.cam_id}_manual_test_position_marker_fake.h5')
            else:
                output_filepath = output_filepath.parent.joinpath(f'{self.cam_id}_manual_test_position_marker_fake.h5')
        df_out = self._construct_dlc_output_style_df_from_manual_marker_coords()
        df_out.to_hdf(output_filepath, "df")
        print(f'Your dataframe was successfully saved at: {output_filepath.as_posix()}.')
        self.load_test_position_markers_df_from_dlc_prediction(filepath_deeplabcut_prediction = output_filepath)
        print('Your "fake DLC marker perdictions" were successfully loaded to the SingleCamDataForAnipose object!')


    def validate_test_position_marker_ids(self, test_positions_gt: TestPositionsGroundTruth, add_missing_marker_ids_with_0_likelihood: bool=True) -> None:
        if hasattr(self, 'test_position_markers_df') == False:
            raise ValueError('There was no DLC prediction of the test position markers loaded yet. '
                             'Please load it using the ".load_test_position_markers_df_from_dlc_prediction()" '
                             'method on this object (if you have DLC predictions to load) - or first add '
                             'the positions manually using the ".add_manual_test_position_marker()" method '
                             'on this object, and eventually load these data after adding all marker_ids '
                             'that you could identify via the ".save_manual_marker_coords_as_fake_dlc_output() '
                             'method on this object.')
        ground_truth_marker_ids = test_positions_gt.unique_marker_ids.copy()
        prediction_marker_ids = list(set([marker_id for scorer, marker_id, key in self.test_position_markers_df.columns]))
        marker_ids_not_in_ground_truth = self._find_non_matching_marker_ids(prediction_marker_ids, ground_truth_marker_ids)
        marker_ids_not_in_prediction = self._find_non_matching_marker_ids(ground_truth_marker_ids, prediction_marker_ids)
        if add_missing_marker_ids_with_0_likelihood & (len(marker_ids_not_in_prediction) > 0):
            self._add_missing_marker_ids_to_prediction(missing_marker_ids = marker_ids_not_in_prediction)
            print('The following marker_ids were missing and added to the dataframe with a '
                  f'likelihood of 0: {marker_ids_not_in_prediction}.')
        if len(marker_ids_not_in_ground_truth) > 0:
            self._remove_marker_ids_not_in_ground_truth(marker_ids_to_remove = marker_ids_not_in_ground_truth)
            print('The following marker_ids were deleted from the dataframe, since they were '
                  f'not present in the ground truth: {marker_ids_not_in_ground_truth}.')
    
    
    def _add_missing_marker_ids_to_prediction(self, missing_marker_ids: List[str]) -> None:
        df = self.test_position_markers_df
        scorer = list(df.columns)[0][0]
        for marker_id in missing_marker_ids:
            for key in ['x', 'y', 'likelihood']:
                df[(scorer, marker_id, key)] = 0


    def _adjust_intrinsic_calibration(self, unadjusted_intrinsic_calibration: Dict) -> Dict:
        adjusted_intrinsic_calibration = unadjusted_intrinsic_calibration.copy()
        # is the following the correct size? current "size" value was determined on grayscale image
        intrinsic_calibration_video_size = unadjusted_intrinsic_calibration['size']
        new_video_size = self._get_anipose_calibration_video_size()
        x_offset, y_offset = self._get_correct_x_y_offsets(intrinsic_calibration_video_size = intrinsic_calibration_video_size, new_video_size = new_video_size)
        adjusted_K = self._get_adjusted_K(K = unadjusted_intrinsic_calibration['K'], x_offset = x_offset, y_offset = y_offset)
        adjusted_intrinsic_calibration = self._incorporate_adjustments_in_intrinsic_calibration(intrinsic_calibration = unadjusted_intrinsic_calibration.copy(),
                                                                                                new_size = new_video_size,
                                                                                                adjusted_K = adjusted_K)
        return adjusted_intrinsic_calibration


    def _check_if_all_details_were_provided_otherwise_set_defaults(self) -> None:
        attributes_and_default_value_setters = {'cropping_offsets': self.add_cropping_offsets,
                                                'flipped_horizontally': self.add_flipping_details, 
                                                'degrees_rotated_clockwise': self.add_rotation_details}
        for attribute_name, default_setter in attributes_and_default_value_setters.items():
            if hasattr(self, attribute_name) == False:
                default_setter()
                if attribute_name == 'flipped_horizontally':
                    key_to_print = 'flipped_horizontally" & "flipped_vertically'
                else:
                    key_to_print = attribute_name
                print(f'User info: since no other information were provided, "{key_to_print}" '
                      f'were set to the corresponding default values: {getattr(self, attribute_name)}.')


    def _construct_dlc_output_style_df_from_manual_marker_coords(self) -> pd.DataFrame:
        multi_index = self._get_multi_index()
        df = pd.DataFrame(data = {}, columns = multi_index)
        for scorer, marker_id, key in df.columns:
            df[(scorer, marker_id, key)] = self.manual_test_position_marker_coords_pred[marker_id][key]
        return df


    def _find_non_matching_marker_ids(self, marker_ids_to_match: List[str], template_marker_ids: List[str]) -> List:
        return [marker_id for marker_id in marker_ids_to_match if marker_id not in template_marker_ids]


    def _get_adjusted_K(self, K: np.ndarray, x_offset: int, y_offset: int) -> np.ndarray:
        adjusted_K = K.copy()
        adjusted_K[0][2] = adjusted_K[0][2] - x_offset
        adjusted_K[1][2] = adjusted_K[1][2] - y_offset
        return adjusted_K  
    
    
    def _get_anipose_calibration_video_size(self) -> Tuple[int, int]:
        reader = iio.get_reader(self.filepath_synchronized_calibration_video)
        frame = reader.get_data(0)
        return frame.shape[1], frame.shape[0] 
    
    
    def _get_correct_x_y_offsets(self, intrinsic_calibration_video_size: Tuple[int, int], new_video_size: Tuple[int, int]) -> Tuple[int, int]:
        # ToDo:
        # incorporate rotation?
        x_offset, y_offset = self.cropping_offsets
        if self.flipped_vertically:
            x_offset = intrinsic_calibration_video_size[0] - new_video_size[0] - self.cropping_offsets[0]
        if self.flipped_horizontally:
            y_offset = intrinsic_calibration_video_size[1] - new_video_size[1] - self.cropping_offsets[1]
        return x_offset, y_offset

    
    def _get_multi_index(self) -> pd.MultiIndex:
        multi_index_column_names = [[], [], []]
        for marker_id in self.manual_test_position_marker_coords_pred.keys():
            for column_name in ("x", "y", "likelihood"):
                multi_index_column_names[0].append("manually_annotated_marker_positions")
                multi_index_column_names[1].append(marker_id)
                multi_index_column_names[2].append(column_name)
        return pd.MultiIndex.from_arrays(multi_index_column_names, names=('scorer', 'bodyparts', 'coords'))


    def _incorporate_adjustments_in_intrinsic_calibration(self, intrinsic_calibration: Dict, new_size: Tuple[int, int], adjusted_K: np.ndarray) -> Dict:
        intrinsic_calibration['size'] = new_size
        intrinsic_calibration['K'] = adjusted_K
        return intrinsic_calibration


    def _is_adjusting_of_intrinsic_calibration_required(self, unadjusted_intrinsic_calibration: Dict) -> bool:
        self._check_if_all_details_were_provided_otherwise_set_defaults()
        adjusting_required = False
        if any([self.cropping_offsets != (0, 0), self.flipped_horizontally, self.flipped_vertically, self.degrees_rotated_clockwise != 0]):
            adjusting_required = True
        return adjusting_required     
            
            
    def _plot_distorted_and_undistorted_image(self, distorted_image: np.ndarray, undistorted_image: np.ndarray) -> None:
        fig = plt.figure(figsize=(12, 5), facecolor='white')
        gs = fig.add_gridspec(1, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        plt.imshow(distorted_image)
        plt.title('raw image')
        ax2 = fig.add_subplot(gs[0, 1])
        plt.imshow(undistorted_image)
        plt.title('undistorted image based on intrinsic calibration')
        plt.show()


    def _remove_marker_ids_not_in_ground_truth(self, marker_ids_to_remove: List[str]) -> None:
        df = self.test_position_markers_df
        columns_to_remove = [column_name for column_name in df.columns if column_name[1] in marker_ids_to_remove]
        df.drop(columns = columns_to_remove, inplace=True)

    
    def _set_intrinsic_calibration(self, intrinsic_calibration: Dict, adjusting_required: bool) -> None:
        if adjusting_required:
            intrinsic_calibration = self._adjust_intrinsic_calibration(unadjusted_intrinsic_calibration = intrinsic_calibration)
        setattr(self, 'intrinsic_calibration_for_anipose', intrinsic_calibration)
        
        
    def _undistort_fisheye_image_for_inspection(self, image: np.ndarray) -> np.ndarray:
        k_for_fisheye = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.intrinsic_calibration_for_anipose['K'], 
                                                                               self.intrinsic_calibration_for_anipose['D'], 
                                                                               self.intrinsic_calibration_for_anipose['size'], 
                                                                               np.eye(3), 
                                                                               balance=0)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.intrinsic_calibration_for_anipose['K'], 
                                                         self.intrinsic_calibration_for_anipose['D'], 
                                                         np.eye(3), 
                                                         k_for_fisheye, 
                                                         self.intrinsic_calibration_for_anipose['size'], 
                                                         cv2.CV_16SC2)
        return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)  

        
        
        
class CalibrationForAnipose3DTracking:

    def __init__(self, single_cams_to_calibrate: List[SingleCamDataForAnipose]) -> None:
        # ToDo: validate unique filepaths
        self._validate_unique_cam_ids(single_cams_to_calibrate = single_cams_to_calibrate)
        self._validate_test_position_markers_df_is_loaded_to_all_single_cam_objects(single_cams_to_calibrate = single_cams_to_calibrate)
        self.single_cam_objects = single_cams_to_calibrate
        self._get_all_calibration_video_filepaths()
        self._initialize_camera_group()


    @property
    def score_threshold(self) -> float:
        return 0.5
        
    def evaluate_triangulation_of_test_position_markers(self, test_positions_gt: TestPositionsGroundTruth, show_3D_plot: bool=True, verbose: bool=True) -> None:
        self.anipose_io = self._preprocess_dlc_predictions_for_anipose()
        self.anipose_io['p3ds_flat'] = self.camera_group.triangulate(self.anipose_io['points_flat'], progress=True)
        self.anipose_io = self._postprocess_triangulations_and_calculate_reprojection_error(anipose_io = self.anipose_io)
        self.anipose_io = self._add_dataframe_of_triangulated_points(anipose_io = self.anipose_io)
        self.anipose_io = self._add_reprojection_errors_of_all_test_position_markers(anipose_io = self.anipose_io)
        self.anipose_io = self._add_all_real_distances_errors(anipose_io = self.anipose_io, test_positions_gt = test_positions_gt)
        if verbose:
            print(f'Mean reprojection error: {self.anipose_io["reproj_nonan"].mean()}')
            for reference_distance_id, distance_errors in self.anipose_io['distance_errors_in_cm'].items():
                print(f'Using {reference_distance_id} as reference distance, the mean distance error is: {distance_errors["mean_error"]} cm.')
        if show_3D_plot:
            self._show_3D_plot(frame_idx = 0, anipose_io = self.anipose_io, marker_ids_to_connect = test_positions_gt.marker_ids_to_connect_in_3D_plot)
            
            
    def load_calibration(self, filepath: Path) -> None:
        self.camera_group = ap_lib.cameras.CameraGroup.load(filepath)


    def run_calibration(self, use_own_intrinsic_calibration: bool=True, charuco_calibration_board: Optional[ap_lib.boards.CharucoBoard]=None) -> None:
        # ToDo
        # possibility to add verbose=False in calibrate_videos() call to avoid lengthy output?
        # confirm type hinting 
        if charuco_calibration_board == None:
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
            charuco_calibration_board = ap_lib.boards.CharucoBoard(7, 5, square_length=1, marker_length=0.8, marker_bits=6, aruco_dict=aruco_dict)
        self.camera_group.calibrate_videos(videos = self.calibration_video_filepaths, 
                                           board = charuco_calibration_board,
                                           init_intrinsics = not use_own_intrinsic_calibration, 
                                           init_extrinsics = True)


    def save_calibration(self, filepath: Path) -> None:
        # ToDo
        # validate filepath and extension (.toml)
        # and add default alternative
        self.camera_group.dump(filepath)
            
     
    def _add_additional_information_and_continue_preprocessing(self, anipose_io: Dict) -> Dict: 
        n_cams, anipose_io['n_points'], anipose_io['n_joints'], _ = anipose_io['points'].shape
        anipose_io['points'][anipose_io['scores'] < self.score_threshold] = np.nan
        anipose_io['points_flat'] = anipose_io['points'].reshape(n_cams, -1, 2)
        anipose_io['scores_flat'] = anipose_io['scores'].reshape(n_cams, -1)
        return anipose_io


    def _add_all_real_distances_errors(self, anipose_io: Dict, test_positions_gt: TestPositionsGroundTruth) -> Dict:
        all_distance_to_cm_conversion_factors = self._get_conversion_factors_from_different_references(anipose_io = anipose_io, test_positions_gt = test_positions_gt)
        anipose_io = self._add_distances_in_cm_for_each_conversion_factor(anipose_io = anipose_io, conversion_factors = all_distance_to_cm_conversion_factors)
        anipose_io = self._add_distance_errors(anipose_io = anipose_io, gt_distances = test_positions_gt.marker_ids_with_distances)
        return anipose_io


    def _add_dataframe_of_triangulated_points(self, anipose_io: Dict) -> Dict:
        all_points_raw = anipose_io['points']
        all_scores = anipose_io['scores']
        _cams, n_frames, n_joints, _ = all_points_raw.shape
        points_3d = anipose_io['p3ds_flat']
        errors = anipose_io['reprojerr_flat']
        good_points = ~np.isnan(all_points_raw[:, :, :, 0])
        num_cams = np.sum(good_points, axis=0).astype('float')
        all_points_3d = points_3d.reshape(n_frames, n_joints, 3)
        all_errors = errors.reshape(n_frames, n_joints)        
        all_scores[~good_points] = 2
        scores_3d = np.min(all_scores, axis=0)
        scores_3d[num_cams < 2] = np.nan
        all_errors[num_cams < 2] = np.nan
        num_cams[num_cams < 2] = np.nan
        all_points_3d_adj = all_points_3d
        M = np.identity(3)
        center = np.zeros(3)
        df = pd.DataFrame()
        for bp_num, bp in enumerate(anipose_io['bodyparts']):
            for ax_num, axis in enumerate(['x','y','z']):
                df[bp + '_' + axis] = all_points_3d_adj[:, bp_num, ax_num]
            df[bp + '_error'] = anipose_io['reprojerr'][:, bp_num]
            #dout[bp + '_ncams'] = n_cams2[:, bp_num]
            df[bp + '_score'] = scores_3d[:, bp_num]
        for i in range(3):
            for j in range(3):
                df['M_{}{}'.format(i, j)] = M[i, j]
        for i in range(3):
            df['center_{}'.format(i)] = center[i]
        df['fnum'] = np.arange(n_frames)
        anipose_io['df_xyz'] = df
        return anipose_io


    def _add_distance_errors(self, anipose_io: Dict, gt_distances: Dict) -> Dict:
        anipose_io['distance_errors_in_cm'] = {}
        for reference_distance_id, triangulated_distances in anipose_io['distances_in_cm'].items():
            anipose_io['distance_errors_in_cm'][reference_distance_id] = {}
            marker_ids_with_distance_error = self._compute_differences_between_triangulated_and_gt_distances(triangulated_distances = triangulated_distances,
                                                                                                             gt_distances = gt_distances)            
            all_distance_errors = [distance_error for marker_id_a, marker_id_b, distance_error in marker_ids_with_distance_error]
            mean_distance_error = np.asarray(all_distance_errors).mean()
            anipose_io['distance_errors_in_cm'][reference_distance_id] = {'individual_errors': marker_ids_with_distance_error,
                                                                          'mean_error': mean_distance_error}
        return anipose_io
    
    
    def _add_distances_in_cm_for_each_conversion_factor(self, anipose_io: Dict, conversion_factors: Dict) -> Dict:
        anipose_io['distances_in_cm'] = {}
        for reference_distance_id, conversion_factor in conversion_factors.items():
            anipose_io['distances_in_cm'][reference_distance_id] = self._convert_all_xyz_distances(anipose_io = anipose_io, conversion_factor = conversion_factor)
        return anipose_io


    def _add_reprojection_errors_of_all_test_position_markers(self, anipose_io: Dict) -> Dict:
        anipose_io['reprojection_errors_test_position_markers'] = {}
        all_reprojection_errors = []
        for key in anipose_io['df_xyz'].iloc[0].keys():
            if "error" in key:
                reprojection_error = anipose_io['df_xyz'][key].iloc[0]
                marker_id = key[:key.find('_error')]
                anipose_io['reprojection_errors_test_position_markers'][marker_id] = reprojection_error # since we only have a single image
                if type(reprojection_error) != np.nan: 
                    # ToDo:
                    # confirm that it would actually be a numpy nan
                    # or as alternative, use something like this after blindly appending all errors to drop the nan´s:
                    # anipose_io['reprojerr'][np.logical_not(np.isnan(anipose_io['reprojerr']))]
                    all_reprojection_errors.append(reprojection_error)
        anipose_io['reprojection_errors_test_position_markers']['mean'] = np.asarray(all_reprojection_errors).mean()
        return anipose_io 


    def _compute_differences_between_triangulated_and_gt_distances(self, triangulated_distances: Dict, gt_distances: Dict) -> List[Tuple[str, str, float]]:
        marker_ids_with_distance_error = []
        for marker_id_a in triangulated_distances.keys():
            for marker_id_b in triangulated_distances[marker_id_a].keys():
                if (marker_id_a in gt_distances.keys()) & (marker_id_b in gt_distances[marker_id_a].keys()):
                    gt_distance = gt_distances[marker_id_a][marker_id_b]
                    triangulated_distance = triangulated_distances[marker_id_a][marker_id_b]
                    distance_error = gt_distance - triangulated_distance
                    marker_ids_with_distance_error.append((marker_id_a, marker_id_b, distance_error))
        return marker_ids_with_distance_error
    
    
    def _connect_all_marker_ids(self, ax: plt.Figure, points: np.ndarray, scheme: List[Tuple[str]], bodyparts: List[str]) -> List[plt.Figure]:
        # ToDo: correct type hints
        cmap = plt.get_cmap('tab10')
        bp_dict = dict(zip(bodyparts, range(len(bodyparts))))
        lines = []
        for i, bps in enumerate(scheme):
            line = self._connect_one_set_of_marker_ids(ax = ax, points = points, bps = bps, bp_dict = bp_dict, color = cmap(i)[:3])
            lines.append(line)
        return lines # return neccessary?
    
    
    def _connect_one_set_of_marker_ids(self, ax: plt.Figure, points: np.ndarray, bps: List[str], bp_dict: Dict, color: np.ndarray) -> plt.Figure:
        # ToDo: correct type hints
        ixs = [bp_dict[bp] for bp in bps]
        return ax.plot(points[ixs, 0], points[ixs, 1], points[ixs, 2], color=color)        


    def _convert_all_xyz_distances(self, anipose_io: Dict, conversion_factor: float) -> Dict:
        marker_id_combinations = it.combinations(anipose_io['bodyparts'], 2)
        all_distances_in_cm = {}
        for marker_id_a, marker_id_b in marker_id_combinations:
            if marker_id_a not in all_distances_in_cm.keys():
                all_distances_in_cm[marker_id_a] = {}
            xyz_distance = self._get_xyz_distance_in_triangulation_space(marker_ids = (marker_id_a, marker_id_b), df_xyz = anipose_io['df_xyz'])
            all_distances_in_cm[marker_id_a][marker_id_b] = xyz_distance / conversion_factor
        return all_distances_in_cm    
    
    
    def _get_all_calibration_video_filepaths(self) -> None:
        video_filepaths = [[single_cam.filepath_synchronized_calibration_video.as_posix()] for single_cam in self.single_cam_objects]
        setattr(self, 'calibration_video_filepaths', video_filepaths)
        
                                                                                                      
    def _get_conversion_factors_from_different_references(self, anipose_io: Dict, test_positions_gt: TestPositionsGroundTruth) -> Dict: # Tuple? List?
        all_conversion_factors = {}
        for reference_distance_id, reference_marker_ids in test_positions_gt.reference_distance_ids_with_corresponding_marker_ids:
            distance_in_cm = test_positions_gt.marker_ids_with_distances[reference_marker_ids[0]][reference_marker_ids[1]]
            distance_to_cm_conversion_factor = self._get_xyz_to_cm_conversion_factor(reference_marker_ids = reference_marker_ids, 
                                                                                     distance_in_cm = distance_in_cm,
                                                                                     df_xyz = anipose_io['df_xyz'])
            all_conversion_factors[reference_distance_id] = distance_to_cm_conversion_factor
        return all_conversion_factors
    

    def _get_xyz_distance_in_triangulation_space(self, marker_ids: Tuple[str, str], df_xyz: pd.DataFrame) -> float:
        squared_differences = [(df_xyz[f'{marker_ids[0]}_{axis}'] - df_xyz[f'{marker_ids[1]}_{axis}'])**2 for axis in ['x', 'y', 'z']]
        return sum(squared_differences)**0.5

    
    def _get_xyz_to_cm_conversion_factor(self, reference_marker_ids: Tuple[str, str], distance_in_cm: Union[int, float], df_xyz: pd.DataFrame) -> float:
        distance_in_triangulation_space = self._get_xyz_distance_in_triangulation_space(marker_ids = reference_marker_ids, df_xyz = df_xyz)
        return distance_in_triangulation_space / distance_in_cm   
    

    def _initialize_camera_group(self) -> None:
        all_Camera_objects = [single_cam.export_as_aniposelib_Camera_object(single_cam.fisheye) for single_cam in self.single_cam_objects]
        setattr(self, 'camera_group', ap_lib.cameras.CameraGroup(all_Camera_objects))
    

    def _preprocess_dlc_predictions_for_anipose(self) -> Dict:
        fname_dict = {}
        for single_cam in self.single_cam_objects:
            fname_dict[single_cam.cam_id] = single_cam.filepath_test_position_marker_prediction
        anipose_io = ap_lib.utils.load_pose2d_fnames(fname_dict = fname_dict)
        anipose_io = self._add_additional_information_and_continue_preprocessing(anipose_io = anipose_io)
        return anipose_io


    def _postprocess_triangulations_and_calculate_reprojection_error(self, anipose_io: Dict) -> Dict:
        anipose_io['reprojerr_flat'] = self.camera_group.reprojection_error(anipose_io['p3ds_flat'], anipose_io['points_flat'], mean=True)
        anipose_io['p3ds'] = anipose_io['p3ds_flat'].reshape(anipose_io['n_points'], anipose_io['n_joints'], 3)
        anipose_io['reprojerr'] = anipose_io['reprojerr_flat'].reshape(anipose_io['n_points'], anipose_io['n_joints'])
        anipose_io['reproj_nonan'] = anipose_io['reprojerr'][np.logical_not(np.isnan(anipose_io['reprojerr']))]
        return anipose_io


    def _show_3D_plot(self, frame_idx: int, anipose_io: Dict, marker_ids_to_connect: List[Tuple[str]]) -> None:
        p3d = anipose_io['p3ds'][frame_idx]
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(p3d[:,0], p3d[:,1], p3d[:,2], c='black', s=100)
        self._connect_all_marker_ids(ax = ax, points = p3d, scheme = marker_ids_to_connect, bodyparts = anipose_io['bodyparts'])
        for i in range(len(anipose_io['bodyparts'])):
            ax.text(p3d[i,0], p3d[i,1] + 0.01, p3d[i,2], anipose_io['bodyparts'][i], size = 9)
        plt.show()
        

    def _validate_test_position_markers_df_is_loaded_to_all_single_cam_objects(self, single_cams_to_calibrate: List[SingleCamDataForAnipose]):
        # ToDo: call validation on single_cam object (potentially again), to ensure it was done?
        for single_cam in single_cams_to_calibrate:
            if hasattr(single_cam, 'test_position_markers_df') == False:
                raise ValueError('For this evaluation, all SingleCamDataForAnipose objects must have '
                                 'loaded the predicted coordinates of the test position marker ids. '
                                 'However, this data is missing for the SingleCamDataForAnipose object '
                                 f'with the cam_id: {single_cam.cam_id}. Please load it to this object '
                                 'by calling it´s ".load_test_position_markers_df_from_dlc_prediction()" '
                                 'method.')


    def _validate_unique_cam_ids(self, single_cams_to_calibrate: List[SingleCamDataForAnipose]) -> None:
        cam_ids = []
        for single_cam in single_cams_to_calibrate:
            if single_cam.cam_id not in cam_ids:
                cam_ids.append(single_cam.cam_id)
            else:
                raise ValueError(f'You added multiple cameras with the cam_id {single_cam.cam_id}, '
                                 'however, all cam_ids must be unique! Please check for duplicates '
                                 'in the "single_cams_to_calibrate" list, or rename the respective '
                                 'cam_id attribute of the corresponding SingleCamDataForAnipose object.')


    def _get_length_in_3d_space(self, PointA: np.array, PointB: np.array) -> float:
            length = math.sqrt((PointA[0]-PointB[0])**2 + (PointA[1]-PointB[1])**2 + (PointA[2]-PointB[2])**2)
            return length

    def _get_angle_from_law_of_cosines(self, length_a: float, length_b: float, length_c: float)->float:
        cos_angle = (length_a**2 + length_b**2 - length_c**2) / (2 * length_b * length_c)
        return math.degrees(math.acos(cos_angle))

    def _get_angle_between_three_points_at_PointC(self, PointA: np.array, PointB: np.array, PointC: np.array) -> float:
        length_c = self._get_length_in_3d_space(PointA, PointB)
        length_b = self._get_length_in_3d_space(PointA, PointC)
        length_a = self._get_length_in_3d_space(PointB, PointC)
        return self._get_angle_from_law_of_cosines(length_a, length_b, length_c)

    def _get_coordinates_plane_equation_from_three_points(self, PointA: np.array, PointB: np.array, PointC: np.array) -> np.array:
        R1 = self._get_Richtungsvektor_from_two_points(PointA, PointB)
        R2 = self._get_Richtungsvektor_from_two_points(PointA, PointC)
        #check for linear independency
        #np.solve: R2 * x != R1
        plane_equation_coordinates = np.asarray([PointA, R1, R2])
        return plane_equation_coordinates
     
        
    def _get_vector_product(self, A: np.array, B: np.array) -> np.array:
        #Kreuzprodukt
        N = np.asarray([A[1]*B[2] - A[2]*B[1], A[2]*B[0]-A[0]*B[2], A[0]*B[1]-A[1]*B[0]])
        return N

    def _get_Richtungsvektor_from_two_points(self, PointA: np.array, PointB: np.array) -> np.array:
        R = np.asarray([PointA[0] - PointB[0], PointA[1] - PointB[1], PointA[2] - PointB[2]])
        return R

    def _get_vector_length(self, vector: np.array) -> float:
        length = math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
        return length

    def _get_angle_between_plane_and_line(self, N: np.array, R: np.array) -> float:
        cosphi = self._get_vector_length(vector = self._get_vector_product(A = N, B = R)) / (self._get_vector_length(N) * self._get_vector_length(R))
        phi = math.degrees(math.acos(cosphi))
        angle = 90 - phi
        return angle

    def _get_angle_between_two_points_and_plane(self, PointA: np.array, PointB: np.array, N: np.array)->float:
        R = self._get_Richtungsvektor_from_two_points(PointA, PointB)
        return self._get_angle_between_plane_and_line(N = N, R = R)


    def _get_distance_between_plane_and_point(self, N: np.array, PointOnPlane: np.array, DistantPoint: np.array)->float:
        a = N[0] * PointOnPlane [0] + N[1] * PointOnPlane [1] + N[2]* PointOnPlane [2]
        distance = abs(N[0] * DistantPoint[0] + N[1] * DistantPoint[1] + N[2] * DistantPoint[2] - a)/math.sqrt(N[0]**2 + N[1]**2 + N[2]**2)
        return distance

    def _get_vector_from_label(self, label: str)->np.array:
        return np.asarray([self.anipose_io['df_xyz'][label + '_x'], self.anipose_io['df_xyz'][label + '_y'], self.anipose_io['df_xyz'][label + '_z']])
    
    
    def run_calibration_control(self, show_full_output = False)->None:
        #check for tilt in maze_plane
        #mazecorners have 90 degrees
        maze_corner_closed_left = self._get_vector_from_label(label = 'maze_corner_closed_left')
        maze_corner_open_left = self._get_vector_from_label(label = 'maze_corner_open_left')
        maze_corner_closed_right = self._get_vector_from_label(label = 'maze_corner_closed_right')
        maze_corner_open_right = self._get_vector_from_label(label = 'maze_corner_open_right')
        
        angle_at_open_right = self._get_angle_between_three_points_at_PointC(PointA = maze_corner_open_left, PointB = maze_corner_closed_right, PointC = maze_corner_open_right)
        angle_at_open_left = self._get_angle_between_three_points_at_PointC(PointA = maze_corner_open_right, PointB = maze_corner_closed_left, PointC = maze_corner_open_left)
        angle_at_closed_right = self._get_angle_between_three_points_at_PointC(PointA = maze_corner_closed_left, PointB = maze_corner_open_right, PointC = maze_corner_closed_right)
        angle_at_closed_left = self._get_angle_between_three_points_at_PointC(PointA = maze_corner_open_left, PointB = maze_corner_closed_right, PointC = maze_corner_closed_left)
        
        print(f'Maze tilted?:\n\n'
            f'Angle at open right: {angle_at_open_right}\n'
            f'Angle at open left: {angle_at_open_left}\n'
            f'Angle at closed right: {angle_at_closed_right}\n'
            f'Angle at closed left: {angle_at_closed_left}\n\n')

        #check for angle of objects on the plane
        plane_coord = self._get_coordinates_plane_equation_from_three_points(PointA = maze_corner_open_left, PointB = maze_corner_closed_right, PointC = maze_corner_closed_left)
        N = self._get_vector_product(A = plane_coord[0], B = plane_coord[2])
        
        screw1_bottom = self._get_vector_from_label(label = 'screw1_bottom')
        screw1_top = self._get_vector_from_label(label = 'screw1_top')
        angle_screw_1 = self._get_angle_between_two_points_and_plane(PointA = screw1_bottom, PointB = screw1_top, N = N)
        screw2_bottom = self._get_vector_from_label(label = 'screw2_bottom')
        screw2_top = self._get_vector_from_label(label = 'screw2_top')
        angle_screw_2 = self._get_angle_between_two_points_and_plane(PointA = screw2_bottom, PointB = screw2_top, N = N)
        screw3_bottom = self._get_vector_from_label(label = 'screw3_bottom')
        screw3_top = self._get_vector_from_label(label = 'screw3_top')
        angle_screw_3 = self._get_angle_between_two_points_and_plane(PointA = screw3_bottom, PointB = screw3_top, N = N)
        screw4_bottom = self._get_vector_from_label(label = 'screw4_bottom')
        screw4_top = self._get_vector_from_label(label = 'screw4_top')
        angle_screw_4 = self._get_angle_between_two_points_and_plane(PointA = screw4_bottom, PointB = screw4_top, N = N)
        
        print(f'Angle of objects on the maze:\n'
            f'Angle screw1: {angle_screw_1}\n'
            f'Angle screw2: {angle_screw_2}\n'
            f'Angle screw3: {angle_screw_3}\n'
            f'Angle screw4: {angle_screw_4}\n')
            
        #X1/X2 are close to the plane
        x1 = self._get_vector_from_label(label = 'x1')
        x2 = self._get_vector_from_label(label = 'x2')
        
        distance_X1_to_maze_plane = self._get_distance_between_plane_and_point(N = N, PointOnPlane = plane_coord[0], DistantPoint = x1)[0]
        distance_X2_to_maze_plane = self._get_distance_between_plane_and_point(N = N, PointOnPlane = plane_coord[0], DistantPoint = x2)[0]
        #in cm?

           
        print(f'Distance of X1 and X2:\n'
        f'Distance X1 to maze: {distance_X1_to_maze_plane}\n'
        f'Distance X2 to maze: {distance_X2_to_maze_plane}\n')