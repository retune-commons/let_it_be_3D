import datetime
import pickle
from pathlib import Path
from typing import List, Tuple, Dict

import imageio as iio
import numpy as np

from .camera_intrinsics import (
    IntrinsicCalibratorFisheyeCamera,
    IntrinsicCalibratorRegularCameraCheckerboard
)
from .user_specific_rules import user_specific_rules_on_videometadata
from .utils import check_keys


class VideoMetadata:
    def __init__(
            self,
            video_filepath: Path,
            recording_config_dict: Dict,
            project_config_dict: Dict,
            tag: str,
    ) -> None:
        self.calvin, self.recording, self.calibration = False, False, False
        if tag in ["calvin", "recording", "calibration"]:
            setattr(self, tag, True)
        else:
            raise KeyError(f"{tag} is not a valid tag for VideoMetadata!\n"
                           f"Only [calvin, recording, calibration] are valid.")
        self.exclusion_state = "valid"
        self.filepath = self._check_filepaths(video_filepath=video_filepath)

        state = self._read_metadata(
            recording_config_dict=recording_config_dict,
            project_config_dict=project_config_dict,
            video_filepath=video_filepath,
        )

        if state == "del":
            raise TypeError("This video_metadata has problems. Use the filename checker to resolve!")
        else:
            self.intrinsic_calibration, self.intrinsic_calibration_filepath = self._get_intrinsic_parameters(
                max_calibration_frames=self.max_calibration_frames,
            )
        if self.calvin:
            self.framenum = 1
        else:
            self.framenum = iio.v2.get_reader(video_filepath).count_frames()

    def _print_message(self, attributes: List) -> None:
        pass

    def _rename_file(self) -> bool:
        pass

    def _check_filepaths(self, video_filepath: Path) -> Path:
        if (video_filepath.suffix in [".mp4", ".mov", ".AVI", ".avi", ".jpg", ".png", ".tiff",
                                      ".bmp"]) and video_filepath.exists():
            return video_filepath
        else:
            raise ValueError("The filepath is not linked to a video or image.")

    def _read_metadata(
            self,
            project_config_dict: Dict,
            recording_config_dict: Dict,
            video_filepath: Path,
    ) -> str:
        for attribute in ["valid_cam_ids", "paradigms", "animal_lines", "load_calibration", "max_calibration_frames",
                          "max_ram_digestible_frames", "max_cpu_cores_to_pool"]:
            setattr(self, attribute, project_config_dict[attribute])
        self.intrinsic_calibration_directory = Path(project_config_dict["intrinsic_calibration_directory"])
        while True:
            undefined_attributes = self._extract_filepath_metadata()
            if undefined_attributes:
                self._print_message(attributes=undefined_attributes)
                delete = self._rename_file()
                if delete:
                    self.filepath.unlink()
                    return "del"
            else:
                break
        self.recording_date = self.recording_date.strftime("%y%m%d")
        if self.recording:
            self.mouse_id = self.mouse_line + "_" + self.mouse_number

        # led pattern not necessary if no synchronisation necessary
        for attribute in ["led_pattern", "target_fps", "calibration_index"]:
            setattr(self, attribute, recording_config_dict[attribute])
        if self.recording_date != recording_config_dict["recording_date"]:
            raise ValueError(
                f"The date of the recording_config_file and the provided video {video_filepath} do not match! Did you pass the right config-file and check the filename carefully?"
            )

        metadata_dict = recording_config_dict[self.cam_id]
        keys_per_cam = ["fps", "offset_row_idx", "offset_col_idx", "flip_h", "flip_v", "fisheye"]
        missing_keys = check_keys(dictionary=metadata_dict, list_of_keys=keys_per_cam)
        if missing_keys:
            raise KeyError(
                f"Missing metadata information in the recording_config_file for {self.cam_id} for {missing_keys}."
            )

        # fps not necessary if all cams have the same fps
        # offsets not necessary if no cropping was performed or use_intrinsic_calibration False
        # fisheye not necessary if no camera is fisheye
        for attribute in keys_per_cam:
            setattr(self, attribute, metadata_dict[attribute])

        self.processing_type = project_config_dict["processing_type"][self.cam_id]
        self.calibration_evaluation_type = project_config_dict["calibration_evaluation_type"][self.cam_id]
        self.processing_filepath = Path(project_config_dict["processing_filepath"][self.cam_id])
        self.calibration_evaluation_filepath = Path(project_config_dict["calibration_evaluation_filepath"][self.cam_id])
        self.led_extraction_type = project_config_dict["led_extraction_type"][self.cam_id]
        self.led_extraction_filepath = project_config_dict["led_extraction_filepath"][self.cam_id]
        return "valid"

    def _extract_filepath_metadata(self) -> List:
        user_specific_rules_on_videometadata(videometadata=self)
        if self.calibration:
            for piece in self.filepath.stem.split("_"):
                for cam in self.valid_cam_ids:
                    if piece.lower() == cam.lower():
                        self.cam_id = cam
                    else:
                        try:
                            self.recording_date = datetime.date(
                                year=int("20" + piece[0:2]),
                                month=int(piece[2:4]),
                                day=int(piece[4:6]),
                            )
                        except ValueError:
                            pass
            for attribute in ["cam_id", "recording_date"]:
                if not hasattr(self, attribute):
                    raise ValueError(
                        f"{attribute} was not found in {self.filepath}! Rename the path manually or use the filename_checker!"
                    )

        elif self.calvin:
            for piece in self.filepath.stem.split("_"):
                for cam in self.valid_cam_ids:
                    if piece.lower() == cam.lower():
                        self.cam_id = cam
                    else:
                        try:
                            self.recording_date = datetime.date(
                                year=int("20" + piece[0:2]),
                                month=int(piece[2:4]),
                                day=int(piece[4:6]),
                            )
                        except ValueError:
                            pass
            for attribute in ["cam_id", "recording_date"]:
                if not hasattr(self, attribute):
                    raise ValueError(
                        f"{attribute} was not found in {self.filepath}! Rename the path manually or use the filename_checker!"
                    )

        elif self.recording:
            for piece in self.filepath.stem.split("_"):
                for cam in self.valid_cam_ids:
                    if piece.lower() == cam.lower():
                        self.cam_id = cam
                    elif piece in self.paradigms:
                        self.paradigm = piece
                    elif piece in self.animal_lines:
                        self.mouse_line = piece
                    elif piece.startswith("F"):
                        sub_pieces = piece.split("-")
                        if len(sub_pieces) == 2:
                            try:
                                int(sub_pieces[1])
                                self.mouse_number = piece
                            except ValueError:
                                pass
                    else:
                        try:
                            self.recording_date = datetime.date(
                                year=int("20" + piece[0:2]),
                                month=int(piece[2:4]),
                                day=int(piece[4:6]),
                            )
                        except ValueError:
                            pass
            for attribute in [
                "cam_id",
                "recording_date",
                "paradigm",
                "mouse_line",
                "mouse_number",
            ]:
                if not hasattr(self, attribute):
                    raise ValueError(
                        f"{attribute} was not found in {self.filepath}! Rename the path manually or use the filename_checker!"
                    )
        return []

    def _get_intrinsic_parameters(
            self,
            max_calibration_frames: int,
    ) -> Tuple[Dict, Path]:
        intrinsic_calibration, intrinsic_calibration_filepath = self._get_filepath_to_intrinsic_calibration_and_read_intrinsic_calibration(
            max_calibration_frames=max_calibration_frames)
        adjusting_required = self._is_adjusting_of_intrinsic_calibration_required()
        if adjusting_required:
            intrinsic_calibration = self._adjust_intrinsic_calibration(
                unadjusted_intrinsic_calibration=intrinsic_calibration
            )
        return intrinsic_calibration, intrinsic_calibration_filepath

    def _get_filepath_to_intrinsic_calibration_and_read_intrinsic_calibration(self,
                                                                              max_calibration_frames: int
                                                                              ) -> Tuple[Dict, Path]:
        if self.fisheye:
            if self.load_calibration:
                try:
                    intrinsic_calibration_filepath = [
                        file
                        for file in self.intrinsic_calibration_directory.iterdir()
                        if file.suffix == ".p" and self.cam_id in file.stem
                    ][0]
                    with open(intrinsic_calibration_filepath, "rb") as io:
                        intrinsic_calibration = pickle.load(io)
                except IndexError:
                    raise FileNotFoundError(
                        f"Could not find a file for an intrinsic calibration .p file for {self.cam_id}.\n"
                        f"It is required having a pickle file including came_id in the intrinsic_calibrations_directory\n"
                        f"({self.intrinsic_calibration_directory}) if you use load_calibration = True\n"
                        f"Use 'load_calibration = False' in project_config to calibrate now!"
                    )
            else:
                try:
                    intrinsic_calibration_checkerboard_video_filepath = [
                        file
                        for file in self.intrinsic_calibration_directory.iterdir()
                        if file.suffix  in [".mp4", ".AVI", ".mov"]
                           and "checkerboard" in file.stem
                           and self.cam_id in file.stem
                    ][0]
                except IndexError:
                    raise FileNotFoundError(
                        f"Could not find a file for a checkerboard video for {self.cam_id}.\n"
                        f"It is required having a checkerboard video .mp4 including checkerboard and cam_id\n"
                        f"in the intrinsic_calibrations_directory ({self.intrinsic_calibration_directory}) if you use load_calibration = False!"
                    )
                calibrator = IntrinsicCalibratorFisheyeCamera(
                    filepath_calibration_video=intrinsic_calibration_checkerboard_video_filepath,
                    max_calibration_frames=max_calibration_frames,
                )
                intrinsic_calibration = calibrator.run()
                intrinsic_calibration_filepath = self.intrinsic_calibration_directory.joinpath(
                    f"checkerboard_intrinsiccalibrationresultsfisheye_{self.cam_id}.p")
                with open(intrinsic_calibration_filepath, "wb") as io:
                    pickle.dump(intrinsic_calibration, io)
        else:
            if self.load_calibration:
                try:
                    intrinsic_calibration_filepath = [
                        file
                        for file in self.intrinsic_calibration_directory.iterdir()
                        if file.suffix == ".p" and self.cam_id in file.stem
                    ][0]
                    with open(intrinsic_calibration_filepath, "rb") as io:
                        intrinsic_calibration = pickle.load(io)
                except IndexError:
                    raise FileNotFoundError(
                        f"Could not find a file for an intrinsic calibration .p file for {self.cam_id}.\n"
                        f"It is required having a pickle file including came_id in the intrinsic_calibrations_directory\n"
                        f"({self.intrinsic_calibration_directory}) if you use load_calibration = True\n‚"
                        f'Use "load_calibration = False" in project_config to calibrate now!'
                    )
            else:
                try:
                    intrinsic_calibration_checkerboard_video_filepath = [
                            file
                            for file in self.intrinsic_calibration_directory.iterdir()
                            if file.suffix in [".mp4", ".AVI", ".mov"]
                               and "checkerboard" in file.stem
                               and self.cam_id in file.stem
                        ][0]
                except IndexError:
                    raise FileNotFoundError(
                        f"Could not find a filepath for an intrinsic calibration or a checkerboard video for {self.cam_id}.\n"
                        f"It is required having a intrinsic_calibration .p file "
                        f"or a checkerboard video in the intrinsic_calibrations_directory ({self.intrinsic_calibration_directory}) for a fisheye-camera!"
                    )
                calibrator = IntrinsicCalibratorRegularCameraCheckerboard(
                    filepath_calibration_video=intrinsic_calibration_checkerboard_video_filepath,
                    max_calibration_frames=max_calibration_frames,
                )
                intrinsic_calibration = calibrator.run()
                intrinsic_calibration_filepath = self.intrinsic_calibration_directory.joinpath(
                    f"checkerboard_intrinsiccalibrationresults_{self.cam_id}.p")
                with open(intrinsic_calibration_filepath, "wb") as io:
                    pickle.dump(intrinsic_calibration, io)
        return intrinsic_calibration, intrinsic_calibration_filepath

    def _is_adjusting_of_intrinsic_calibration_required(self) -> bool:
        adjusting_required = False
        if any(
                [
                    self.offset_col_idx != 0,
                    self.offset_row_idx != 0,
                    self.flip_h,
                    self.flip_v,
                ]
        ):
            adjusting_required = True
        return adjusting_required

    def _adjust_intrinsic_calibration(
            self, unadjusted_intrinsic_calibration: Dict
    ) -> Dict:
        intrinsic_calibration_video_size = unadjusted_intrinsic_calibration["size"]
        new_video_size = self._get_cropped_video_size()
        self.offset_row_idx, self.offset_col_idx = self._get_correct_x_y_offsets(
            intrinsic_calibration_video_size=intrinsic_calibration_video_size,
            new_video_size=new_video_size,
            offset_col_idx=self.offset_col_idx,
            offset_row_idx=self.offset_row_idx
        )
        adjusted_K = self._get_adjusted_K(K=unadjusted_intrinsic_calibration["K"])
        adjusted_intrinsic_calibration = (
            self._incorporate_adjustments_in_intrinsic_calibration(
                intrinsic_calibration=unadjusted_intrinsic_calibration.copy(),
                new_size=new_video_size,
                adjusted_K=adjusted_K,
            )
        )
        return adjusted_intrinsic_calibration

    def _get_cropped_video_size(self) -> Tuple[int, int]:
        try:
            size = iio.v3.immeta(self.filepath, exclude_applied=False)["size"]
        except KeyError:
            size = iio.v3.immeta(self.filepath, exclude_applied=False)["shape"]
        return size

    def _get_correct_x_y_offsets(
            self,
            intrinsic_calibration_video_size: Tuple[int, int],
            new_video_size: Tuple[int, int],
            offset_row_idx: int,
            offset_col_idx: int,
    ) -> Tuple[int, int]:
        if self.flip_v:
            offset_row_idx = (
                    intrinsic_calibration_video_size[1]
                    - new_video_size[0]
                    - offset_row_idx
            )
        if self.flip_h:
            offset_col_idx = (
                    intrinsic_calibration_video_size[0]
                    - new_video_size[1]
                    - offset_col_idx
            )
        return offset_row_idx, offset_col_idx

    def _get_adjusted_K(self, K: np.ndarray) -> np.ndarray:
        adjusted_K = K.copy()
        adjusted_K[0][2] = adjusted_K[0][2] - self.offset_col_idx
        adjusted_K[1][2] = adjusted_K[1][2] - self.offset_row_idx
        return adjusted_K

    def _incorporate_adjustments_in_intrinsic_calibration(
            self,
            intrinsic_calibration: Dict,
            new_size: Tuple[int, int],
            adjusted_K: np.ndarray,
    ) -> Dict:
        intrinsic_calibration["size"] = new_size
        intrinsic_calibration["K"] = adjusted_K
        return intrinsic_calibration


class VideoMetadataChecker(VideoMetadata):
    def _extract_filepath_metadata(self) -> List[str]:
        undefined_attributes = []
        user_specific_rules_on_videometadata(videometadata=self)
        if self.calibration:
            for piece in self.filepath.stem.split("_"):
                for cam in self.valid_cam_ids:
                    if piece.lower() == cam.lower():
                        self.cam_id = cam
                    else:
                        try:
                            self.recording_date = datetime.date(
                                year=int("20" + piece[0:2]),
                                month=int(piece[2:4]),
                                day=int(piece[4:6]),
                            )
                        except ValueError:
                            pass

            for attribute in ["cam_id", "recording_date"]:
                if not hasattr(self, attribute):
                    undefined_attributes.append(attribute)

        elif self.calvin:
            for piece in self.filepath.stem.split("_"):
                for cam in self.valid_cam_ids:
                    if piece.lower() == cam.lower():
                        self.cam_id = cam
                    else:
                        try:
                            self.recording_date = datetime.date(
                                year=int("20" + piece[0:2]),
                                month=int(piece[2:4]),
                                day=int(piece[4:6]),
                            )
                        except ValueError:
                            pass

            for attribute in ["cam_id", "recording_date"]:
                if not hasattr(self, attribute):
                    undefined_attributes.append(attribute)

        elif self.recording:
            for piece in self.filepath.stem.split("_"):
                for cam in self.valid_cam_ids:
                    if piece.lower() == cam.lower():
                        self.cam_id = cam
                    elif piece in self.paradigms:
                        self.paradigm = piece
                    elif piece in self.animal_lines:
                        self.mouse_line = piece
                    elif piece.startswith("F"):
                        sub_pieces = piece.split("-")
                        if len(sub_pieces) == 2:
                            try:
                                int(sub_pieces[1])
                                self.mouse_number = piece
                            except ValueError:
                                pass
                    else:
                        try:
                            self.recording_date = datetime.date(
                                year=int("20" + piece[0:2]),
                                month=int(piece[2:4]),
                                day=int(piece[4:6]),
                            )
                        except ValueError:
                            pass

            for attribute in [
                "cam_id",
                "recording_date",
                "paradigm",
                "mouse_line",
                "mouse_number",
            ]:
                if not hasattr(self, attribute):
                    undefined_attributes.append(attribute)

        return undefined_attributes

    def _print_message(self, attributes: List[str]) -> None:
        print(
            f"The information {attributes} could not be extracted automatically from the following file:\n"
            f"{self.filepath}"
        )
        for attribute in attributes:
            if attribute == "cam_id":
                print(
                    f"Cam_id was not found in filename or did not match any of the defined cam_ids. \nPlease include one of the following ids into the filename: {self.valid_cam_ids} or add the cam_id to valid_cam_ids!"
                )
            elif attribute == "recording_date":
                print(
                    f"Recording_date was not found in filename or did not match the required structure for date. \nPlease include the date as YYMMDD , e.g., 220928, into the filename!"
                )
            elif attribute == "paradigm":
                f"Paradigm was not found in filename or did not match any of the defined paradigms. \nPlease Please include one of the following paradigms into the filename: {self.paradigms} or add the paradigm to paradigms!"
            elif attribute == "mouse_line":
                print(
                    f"Mouse_line was not found in filename or is not supported. \nPlease include one of the following lines into the filename: {self.animal_lines} or add the line to valid_mouse_lines!"
                )
            elif attribute == "mouse_number":
                print(
                    "Mouse_number was not found in filename or did not match the required structure for a mouse_number. \n Please include the mouse number as Generation-Number, e.g., F12-45, into the filename!"
                )

    def _rename_file(self) -> bool:
        suffix = self.filepath.suffix
        new_filename = input(
            f"Enter new filename! \nIf the video is invalid, enter x and it will be deleted!\n If the video belongs to another folder, enter y, and move it manually!\n{self.filepath.parent}/"
        )
        if new_filename == "y":
            print(f"{self.filepath} needs to be moved!")
            raise TypeError
        if new_filename == "x":
            return True
        new_filepath = self.filepath.parent.joinpath(
            Path(new_filename).with_suffix(suffix)
        )
        if new_filepath == self.filepath:
            print("The entered filename and the real filename are identical.")
        elif new_filepath.exists():
            print(
                "Couldn't rename file, since the entered filename does already exist."
            )
        else:
            self.filepath.rename(new_filepath)
            self.filepath = new_filepath
        return False
