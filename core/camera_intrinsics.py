from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Dict, Union

import cv2
import imageio as iio
import numpy as np
from numpy import ndarray

from .utils import convert_to_path


class IntrinsicCameraCalibrator(ABC):
    """
    Intrinsic calibration for fisheye and regular cameras on checkerboard videos.

    Parameters
    __________
    filepath_calibration_video: Path or str
        The filepath to the intrinsic calibration videos. They have to be
        recorded in same resolution as the recording/calibration videos
        without cropping using a 6x6 checkerboard..
    max_calibration_frames: int
        Number of frames to take into account for intrinsic calibration.
        300 works well, depending on CPU speed, it can be necessary to reduce.

    Attributes
    __________
    filepath_calibration_video: Path or str
        The filepath to the intrinsic calibration videos. They have to be
        recorded in same resolution as the recording/calibration videos
        without cropping using a 6x6 checkerboard..
    max_calibration_frames: int
        Number of frames to take into account for intrinsic calibration.
        300 works well, depending on CPU speed, it can be necessary to reduce.
    video_reader: Reader
        imageio Reader object of the calibration video.
    checkerboard_rows_and_columns
    d: np.ndarray
        Empty camera matrix.
    imsize: tuple of ints
        Size of the video.
    k: np.ndarray
        Empty camera matrix.
    objp: np.ndarray
        Empty object points. Shape as detected in one frame.

    Methods
    _______
    run()
        Detect board in frames and calibrate intrinsics.

    References
    __________
    [1] Kenneth Jiang (2017).
    Calibrate fisheye lens using OpenCV.
    medium.com (https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0)

    Copyright Kenneth Jiang.
    This class and its subclasses use code taken from [1]. Changes were made to
    match our needs here.

    Examples
    ________
    >>> from core.camera_intrinsics import IntrinsicCalibratorFisheyeCamera
    >>> intrinsic_calibration_object = IntrinsicCalibratorFisheyeCamera(
        ... "test_data/intrinsic_calibrations/Bottom_checkerboard.mp4",
        ... 100)
    >>> intrinsic_calibration_object.run()
    """
    @abstractmethod
    def _run_camera_type_specific_calibration(
            self, objpoints: List[np.ndarray], imgpoints: List[np.ndarray]
    ) -> Tuple:
        # wrapper to camera type specific calibration function
        # all remaining data is stored in attributes of the object
        pass

    @abstractmethod
    def _detect_board_corners(self, frame_idxs: List[int]) -> List[np.ndarray]:
        pass

    @property
    def checkerboard_rows_and_columns(self) -> Tuple[int, int]:
        return 5, 5

    @property
    def d(self) -> np.ndarray:
        return np.zeros((4, 1))

    @property
    def imsize(self) -> Tuple[int, int]:
        frame = np.asarray(self.video_reader.get_data(0))
        frame_in_gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame_in_gray_scale.shape[::-1]

    @property
    def k(self) -> np.ndarray:
        return np.zeros((3, 3))

    @property
    def objp(self) -> np.ndarray:
        objp = np.zeros(
            (
                1,
                self.checkerboard_rows_and_columns[0]
                * self.checkerboard_rows_and_columns[1],
                3,
            ),
            np.float32,
        )
        objp[0, :, :2] = np.mgrid[
                         0: self.checkerboard_rows_and_columns[0],
                         0: self.checkerboard_rows_and_columns[1],
                         ].T.reshape(-1, 2)
        return objp

    def __init__(
            self, filepath_calibration_video: Union[Path, str], max_calibration_frames: int
    ) -> None:
        """
        Construct all necessary attributes for the IntrinsicCameraCalibrator class.

        Parameters
        ----------
        filepath_calibration_video: Path or str
            The filepath to the intrinsic calibration videos. They have to be
            recorded in same resolution as the recording/calibration videos
            without cropping, using a 6x6 checkerboard.
        max_calibration_frames: int
            Number of frames to take into account for intrinsic calibration.
            300 works well, depending on CPU speed, it can be necessary to reduce.
        """
        self.video_filepath = convert_to_path(filepath_calibration_video)
        self.max_calibration_frames = max_calibration_frames
        self.video_reader = iio.v2.get_reader(filepath_calibration_video)

    def run(self) -> Dict:
        """
        Detect board in frames and calibrate intrinsics.

        Returns
        -------
        calibration_results: dict
            Intrinsic calibration results containing camera matrix and
            distorsion coefficient at keys 'K' and 'D'.
        """
        selected_frame_idxs = self._get_indices_of_selected_frames()
        detected_board_corners_per_image = self._detect_board_corners(
            frame_idxs=selected_frame_idxs
        )

        if len(detected_board_corners_per_image) != self.max_calibration_frames:
            detected_board_corners_per_image = (
                self._attempt_to_match_max_frame_count(
                    corners_per_image=detected_board_corners_per_image,
                    already_selected_frame_idxs=selected_frame_idxs,
                )
            )
        object_points = self._compute_object_points(
            n_detected_boards=len(detected_board_corners_per_image)
        )

        retval, K, D, rvec, tvec = self._run_camera_type_specific_calibration(
            objpoints=object_points, imgpoints=detected_board_corners_per_image
        )
        calibration_results = self._construct_calibration_results(
            K=K, D=D
        )
        return calibration_results

    def _get_indices_of_selected_frames(self) -> List[int]:
        sampling_rate = self._determine_sampling_rate()
        frame_idxs = self._get_sampling_frame_indices(sampling_rate=sampling_rate)
        return list(frame_idxs)

    def _attempt_to_match_max_frame_count(
            self,
            corners_per_image: List[np.ndarray],
            already_selected_frame_idxs: List[int],
    ) -> List[np.ndarray]:
        print(f"Frames with detected checkerboard: {len(corners_per_image)}.")
        if len(corners_per_image) < self.max_calibration_frames:
            print("Trying to find some more ...")
            corners_per_image = self._attempt_to_reach_max_frame_count(
                corners_per_image=corners_per_image,
                already_selected_frame_idxs=already_selected_frame_idxs,
            )
            print(
                f"Done. Now we are at a total of {len(corners_per_image)} "
                f"frames in which I could detect a checkerboard."
            )
        elif len(corners_per_image) > self.max_calibration_frames:
            corners_per_image = self._limit_to_max_frame_count(
                all_detected_corners=corners_per_image
            )
            print(f"Limited them to only {len(corners_per_image)}.")
        return corners_per_image

    def _compute_object_points(self, n_detected_boards: int) -> List[np.ndarray]:
        object_points = []
        for i in range(n_detected_boards):
            object_points.append(self.objp)
        return object_points

    def _attempt_to_reach_max_frame_count(
            self,
            corners_per_image: List[np.ndarray],
            already_selected_frame_idxs: List[int],
    ) -> List[np.ndarray]:
        # ToDo: limit time?
        total_frame_count = self.video_reader.count_frames()
        for idx in range(total_frame_count):
            if len(corners_per_image) < self.max_calibration_frames:
                if idx not in already_selected_frame_idxs:
                    (
                        checkerboard_detected,
                        predicted_corners,
                    ) = self._run_checkerboard_corner_detection(idx=idx)
                    if checkerboard_detected:
                        corners_per_image.append(predicted_corners)
            else:
                break
        return corners_per_image

    def _determine_sampling_rate(self) -> int:
        total_frame_count = self.video_reader.count_frames()
        if total_frame_count >= 5 * self.max_calibration_frames:
            sampling_rate = total_frame_count // (5 * self.max_calibration_frames)
        else:
            sampling_rate = 1
        return sampling_rate

    def _get_sampling_frame_indices(self, sampling_rate: int) -> np.ndarray:
        total_frame_count = self.video_reader.count_frames()
        n_frames_to_select = total_frame_count // sampling_rate
        return np.linspace(
            0, total_frame_count, n_frames_to_select, endpoint=False, dtype=int
        )

    def _limit_to_max_frame_count(
            self, all_detected_corners: List[np.ndarray]
    ) -> List[np.ndarray]:
        sampling_idxs = np.linspace(
            0,
            len(all_detected_corners),
            self.max_calibration_frames,
            endpoint=False,
            dtype=int,
        )
        sampled_corners = np.asarray(all_detected_corners)[sampling_idxs]
        return list(sampled_corners)

    def _construct_calibration_results(self, K: np.ndarray, D: np.ndarray) -> Dict:
        calibration_results = {"K": K, "D": D, "size": self.imsize}
        setattr(self, "calibration_results", calibration_results)
        return calibration_results


class IntrinsicCameraCalibratorCheckerboard(IntrinsicCameraCalibrator, ABC):
    def _detect_board_corners(self, frame_idxs: List[int]) -> List[np.ndarray]:
        detected_checkerboard_corners_per_image = []
        for idx in frame_idxs:
            (
                checkerboard_detected,
                predicted_corners,
            ) = self._run_checkerboard_corner_detection(idx=idx)
            if checkerboard_detected:
                detected_checkerboard_corners_per_image.append(predicted_corners)
        return detected_checkerboard_corners_per_image

    def _run_checkerboard_corner_detection(self, idx: int) -> Tuple[bool, np.ndarray]:
        image = np.asarray(self.video_reader.get_data(idx))
        gray_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        checkerboard_detected, predicted_corners = cv2.findChessboardCorners(
            gray_scale_image,
            self.checkerboard_rows_and_columns,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        return checkerboard_detected, predicted_corners


class IntrinsicCalibratorFisheyeCamera(IntrinsicCameraCalibratorCheckerboard):
    def _compute_rvecs_and_tvecs(
            self, n_detected_boards: int
    ) -> Tuple[List[ndarray], List[ndarray]]:
        rvecs = [
            np.zeros((1, 1, 3), dtype=np.float64) for i in range(n_detected_boards)
        ]
        tvecs = [
            np.zeros((1, 1, 3), dtype=np.float64) for i in range(n_detected_boards)
        ]
        return rvecs, tvecs

    def _run_camera_type_specific_calibration(
            self, objpoints: List[np.ndarray], imgpoints: List[np.ndarray]
    ) -> Tuple:
        rvecs, tvecs = self._compute_rvecs_and_tvecs(n_detected_boards=len(objpoints))
        calibration_flags = (
                cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                + cv2.fisheye.CALIB_CHECK_COND
                + cv2.fisheye.CALIB_FIX_SKEW
        )
        new_subpixel_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            1e-6,
        )
        return cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            self.imsize,
            self.k,
            self.d,
            rvecs,
            tvecs,
            calibration_flags,
            new_subpixel_criteria,
        )


class IntrinsicCalibratorRegularCameraCheckerboard(
    IntrinsicCameraCalibratorCheckerboard
):
    def _run_camera_type_specific_calibration(
            self, objpoints: List[np.ndarray], imgpoints: List[np.ndarray]
    ) -> Tuple:
        return cv2.calibrateCamera(objpoints, imgpoints, self.imsize, None, None)
