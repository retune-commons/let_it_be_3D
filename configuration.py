"""
To do:

    rewrite RecordingConfig class to get rid of inputs -> change to function variables
    GUI instead of function-input
    get booleans instead of str from RecordingConfigs flipping and fisheye
    improve led-pattern type

"""


from abc import ABC, abstractmethod
from pathlib import Path
import yaml


class Configs(ABC):
    """Generate Config files (.yaml) from user input."""

    @abstractmethod
    def write_config(self):
        """ write extracted config information in a yaml file"""
        pass


class ProjectConfigs(Configs):
    """Configs subclass for information about the whole project

    Input: target_fps, intristic calibration_dir, animal_lines, paradigms, camID processing_type, processing_path, calibration_evaluation_type, calibration_evaluation_path
    Output: saves let_it_be_3D_project_config in cwd (includes all Inputs and optionally list of cams -> valid_cam_IDs)
    """

    def get_user_input(
        self,
        target_fps: int,
        intrinsic_calibration_dir: str,
        animal_lines: list,
        paradigms: list,
    ):

        user_input = {
            "target_fps": target_fps,
            "intrinsic_calibration_dir": intrinsic_calibration_dir,
            "animal_lines": animal_lines,
            "paradigms": paradigms,
            "processing_type": {},
            "processing_path": {},
            "calibration_evaluation_type": {},
            "calibration_evaluation_path": {},
        }

    def add_camera(
        self,
        camID: str,
        processing_type: str,
        processing_path: str,
        calibration_evaluation_type: str,
        calibration_evaluation_path: str,
    ):

        self.user_input["processing_type"][camID] = processing_type
        self.user_input["processing_path"][camID] = processing_path
        self.user_input["calibration_evaluation_type"][
            camID
        ] = calibration_evaluation_type
        self.user_input["calibration_evaluation_path"][
            camID
        ] = calibration_evaluation_path

    def create_list_of_cameras_used(self):
        self.user_input["valid_cam_IDs"] = self.user_input["processing_type"].keys()

    def write_config(self):
        with open("let_it_be_3D_project_config.yaml", "w") as write:
            yaml.dump(self.user_input, write)


class RecordingConfigs(Configs):
    """Configs subclass for information about the individual recording day
    input: path of the project config file, which is named
    UI-input: recording_date, led_pattern, fps, offset_row_idx, offset_col_idx, fliph, flipv, fisheye
    output:
    
    """

    def load_projectconfig(self, path_project_config=None):

        try:
            with open(path_project_config, "r") as file:
                self.project_config = yaml.safe_load(file)
        except FileNotFoundError:
            print(
                " Oooops! Project config file not found. Please check the path_project_config variable you put into this function"
            )

    def user_input(self):
        user_input = {}
        user_input["recording_date"] = input(
            "Please write the recording date in the yymmdd format"
        )
        user_input["led_pattern"] = input("What´s the LED pattern?")

        print(
            "The following cameras have been found:",
            self.project_config["valid_cam_IDs"],
        )
        for cam in self.project_config["valid_cam_IDs"]:
            print(
                "\n--------------------------------------------------\n \
            currently working on cameraID: "
                + str(cam)
                + "\
                 \n--------------------------------------------------\n"
            )
            user_input[cam] = {}
            user_input[cam]["fps"] = int(input("fps"))
            user_input[cam]["offset_row_idx"] = int(input("offset row id x"))
            user_input[cam]["offset_col_idx"] = int(input("offset_col_idx"))
            user_input[cam]["fliph"] = input(
                'flip horizontal write "True" for True  and "False" for False'
            )
            user_input[cam]["flipv"] = input(
                'flip vertical write "True" for True  and "False" for False'
            )
            user_input[cam]["fisheye"] = input(
                'fisheye write "True" for True  and "False" for False'
            )

        self.user_input = user_input

    def write_config(self):
        recording_name = (
            "let_it_be_3D_recording_config_"
            + self.user_input["recording_date"]
            + ".yaml"
        )
        with open(recording_name, "w") as write:
            print("\nRecording config successfully written!")
            yaml.dump(self.user_input, write)
