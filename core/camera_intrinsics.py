from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union, Dict

import imageio as iio
import cv2
import numpy as np
import pickle
from pathlib import Path


class IntrinsicCameraCalibrator(ABC):
    # https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0

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
            0 : self.checkerboard_rows_and_columns[0],
            0 : self.checkerboard_rows_and_columns[1],
        ].T.reshape(-1, 2)
        return objp

    @property
    def subpixel_criteria(self) -> Tuple:
        return (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

    def run(self) -> Dict:
        selected_frame_idxs = self._get_indices_of_selected_frames()
        detected_checkerboard_corners_per_image = self._detect_board_corners(
            frame_idxs=selected_frame_idxs
        )

        # check, how the charuco calibration works and whether the following function calls are needed in the abstract class
        # rename to board instead of checkerboard
        if len(detected_checkerboard_corners_per_image) != self.max_frame_count:
            detected_checkerboard_corners_per_image = self._attempt_to_match_max_frame_count(
                corners_per_image=detected_checkerboard_corners_per_image,
                already_selected_frame_idxs=selected_frame_idxs,
            )
        object_points = self._compute_object_points(
            n_detected_boards=len(detected_checkerboard_corners_per_image)
        )
        #

        retval, K, D, rvec, tvec = self._run_camera_type_specific_calibration(
            objpoints=object_points, imgpoints=detected_checkerboard_corners_per_image
        )
        calibration_results = self._construct_calibration_results(
            K=K, D=D, rvec=rvec, tvec=tvec
        )
        return calibration_results

    def _attempt_to_match_max_frame_count(
        self,
        corners_per_image: List[np.ndarray],
        already_selected_frame_idxs: List[int],
    ) -> List[np.ndarray]:
        print(f"Frames with detected checkerboard: {len(corners_per_image)}.")
        if len(corners_per_image) < self.max_frame_count:
            print("Trying to find some more ...")
            corners_per_image = self._attempt_to_reach_max_frame_count(
                corners_per_image=corners_per_image,
                already_selected_frame_idxs=already_selected_frame_idxs,
            )
            print(
                f"Done. Now we are at a total of {len(corners_per_image)} frames in which I could detect a checkerboard."
            )
        elif len(corners_per_image) > self.max_frame_count:
            corners_per_image = self._limit_to_max_frame_count(
                all_detected_corners=corners_per_image
            )
            print(f"Limited them to only {len(corners_per_image)}.")
        return corners_per_image

    def _attempt_to_reach_max_frame_count(
        self,
        corners_per_image: List[np.ndarray],
        already_selected_frame_idxs: List[int],
    ) -> List[np.ndarray]:
        # ToDo
        # limit time?
        total_frame_count = self.video_reader.count_frames()
        for idx in range(total_frame_count):
            if len(corners_per_image) < self.max_frame_count:
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

    def _construct_calibration_results(
        self, K: np.ndarray, D: np.ndarray, rvec: np.ndarray, tvec: np.ndarray
    ) -> Dict:
        # ToDo:
        # - confirm type hints
        calibration_results = {"K": K, "D": D, "size": self.imsize}
        setattr(self, "calibration_results", calibration_results)
        return calibration_results

    def _compute_object_points(self, n_detected_boards: int) -> List[np.ndarray]:
        object_points = []
        for i in range(n_detected_boards):
            object_points.append(self.objp)
        return object_points

    def _determine_sampling_rate(self) -> int:
        total_frame_count = self.video_reader.count_frames()
        if total_frame_count >= 5 * self.max_frame_count:
            sampling_rate = total_frame_count // (5 * self.max_frame_count)
        else:
            sampling_rate = 1
        return sampling_rate

    def _get_indices_of_selected_frames(self) -> List[int]:
        sampling_rate = self._determine_sampling_rate()
        frame_idxs = self._get_sampling_frame_indices(sampling_rate=sampling_rate)
        return list(frame_idxs)

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
            self.max_frame_count,
            endpoint=False,
            dtype=int,
        )
        sampled_corners = np.asarray(all_detected_corners)[sampling_idxs]
        return list(sampled_corners)


class IntrinsicCameraCalibratorCharuco(IntrinsicCameraCalibrator):
    def _detect_board_corners(self, frame_idxs: List[int]) -> List[np.ndarray]:
        pass


class IntrinsicCameraCalibratorCheckerboard(IntrinsicCameraCalibrator):
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
        # if checkerboard_detected:
        # predicted_corners = cv2.cornerSubPix(gray_scale_image, predicted_corners, (3,3), (-1,-1), self.subpixel_criteria)
        return checkerboard_detected, predicted_corners


class IntrinsicCalibratorFisheyeCamera(IntrinsicCameraCalibratorCheckerboard):
    def _compute_rvecs_and_tvecs(
        self, n_detected_boards: int
    ) -> Tuple[np.ndarray, np.ndarray]:
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


class IntrinsicCalibratorRegularCameraCharuco(IntrinsicCameraCalibratorCharuco):
    def _run_camera_type_specific_calibration(self) -> None:
        pass
