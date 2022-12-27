from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union, Dict

import yaml
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from pathlib import Path
import imageio as iio #we need .v2 functions!

from .utils import convert_to_path
from .triangulation_calibration_module import Calibration, Triangulation_Positions, Triangulation_Recordings
from .video_metadata import VideoMetadata

class meta_config():
    def __init__(self, recording_days: List)->None:
        pass
    

class meta_interface(ABC):
    
    def __init__(self, project_config_path: Path)->None:
        self.project_config_path = convert_to_path(project_config_path)
        if not self.project_config_path.exists():
            raise FileNotFoundError("The file doesn't exist. Check your path!")
        self._read_project_config()
        self.recording_configs = []
        self.recording_dates = []
        self.meta = {'project_config_path': str(self.project_config_path), "recording_days": {}}
        
    def _read_project_config(self)->None:
        with open(self.project_config_path, "r") as ymlfile:
            project_config = yaml.load(ymlfile, Loader=yaml.SafeLoader)

        for key in [
            "paradigms",
        ]:
            try:
                project_config[key]
            except KeyError:
                raise KeyError(
                    f"Missing metadata information in the project_config_file {self.project_config_path} for {key}."
                )
        self.paradigms = project_config["paradigms"]
        
    def _read_recording_config(self, recording_config_path: Path)->str:
        with open(recording_config_path, "r") as ymlfile:
            recording_config = yaml.load(ymlfile, Loader=yaml.SafeLoader)

        for key in [
            "recording_date"
        ]:
            try:
                recording_config[key]
            except KeyError:
                raise KeyError(
                    f"Missing metadata information in the recording_config_file {recording_config_path} for {key}."
                )
        self.recording_dates.append(recording_config["recording_date"])
        return str(recording_config["recording_date"])
        
    def select_recording_configs(self)->None:
        Tk().withdraw()
        selected_recording_configs = askopenfilenames(title="Select recording_config.yaml")
        
        for path_to_recording_config in selected_recording_configs:
            self.add_recording_config(path_to_recording_config=path_to_recording_config)
    
    def add_recording_config(self, path_to_recording_config: Path)->None:
        path_to_recording_config = convert_to_path(path_to_recording_config)
        if path_to_recording_config.suffix == ".yaml" and path_to_recording_config.exists():
            self.recording_configs.append(path_to_recording_config)
            recording_date = self._read_recording_config(recording_config_path=path_to_recording_config)
            flag = -1
            for key in self.meta['recording_days'].keys():
                if key.startswith(f'Recording_Day_{recording_date}'):
                    found_flag = int(key[-1:])
                    if found_flag > flag:
                        flag = found_flag
            self.meta['recording_days'][f'Recording_Day_{recording_date}_{str(flag+1)}'] = {'recording_config_path': str(path_to_recording_config), 'recording_date': recording_date, 'recording_directories': [], 'recordings': {}, 'calibrations': {}, 'calibration_directory': str(path_to_recording_config.parent)}
        
    def initialize_meta_config(self)->None:
        for recording_day in self.meta['recording_days'].values():
            for file in Path(recording_day['recording_config_path']).parent.parent.parent.glob("**"):
                if file.name[:len(recording_day['recording_date'])] == recording_day['recording_date'] and file.name[-3:] in self.paradigms:#hardcoded length of paradigm
                    recording_day['recording_directories'].append(str(file))
            recording_day['num_recordings'] = len(recording_day['recording_directories'])
                    
                    
    def add_recording_manually(self, file: Path, recording_day: str)->None:
        file = convert_to_path(file)
        if not file.exists() or recording_day not in self.meta['recording_days'].keys():
            print(f"couldn't add recording directory! \nCheck your filepath and make sure the recording_day is in {self.meta['recording_days'].keys()}!")
        else:
            self.meta['recording_days'][recording_day]['recording_directories'].append(str(file))
            self.meta['recording_days'][recording_day]['num_recordings'] = len(recording_day['recording_directories'])
            print("added recording directory succesfully!")
            
            
    def load_meta_from_yaml(self, filename: Path)->None:
        filename = convert_to_path(filename)
        with open(filename, "r") as ymlfile:
            self.meta = yaml.load(ymlfile, Loader=yaml.SafeLoader)
        
        for recording_day in self.meta['recording_days'].values():
            recording_day['num_recordings'] = len(recording_day['recording_directories'])
        #update class attributes?
    
    def export_meta_to_yaml(self, filename: Path)->None:
        filename = convert_to_path(filename)
        with open(filename, "w") as file:
            yaml.dump(self.meta, file)
            
    def create_recordings(self)->None:
        self.objects = {'triangulation_recordings_objects': {}}
        #optional: create output_directories?
        for recording_day in self.meta["recording_days"]:
            for recording in self.meta["recording_days"][recording_day]['recording_directories']:
                triangulation_recordings_object = Triangulation_Recordings(recording_directory = recording, 
                     calibration_directory = self.meta["recording_days"][recording_day]['calibration_directory'],
                     recording_config_filepath = self.meta["recording_days"][recording_day]['recording_config_path'], 
                     project_config_filepath= self.meta['project_config_path'],
                     output_directory = recording, overwrite = False)
                individual_key = f"{triangulation_recordings_object.mouse_id}_{triangulation_recordings_object.recording_date}_{triangulation_recordings_object.paradigm}"
                videos = {video.cam_id: self._create_video_dict(video=video) for video in triangulation_recordings_object.metadata_from_videos}
                self.meta['recording_days'][recording_day]['recordings'][individual_key] = {'recording_directory': recording,
                                                                            'key': individual_key,
                                                                            'target_fps': triangulation_recordings_object.target_fps,
                                                                            'led_pattern': triangulation_recordings_object.led_pattern,
                                                                            'videos': videos}
                self.objects["triangulation_recordings_objects"][individual_key] = triangulation_recordings_object
            
    def synchronize_recordings(self)->None:
        for recording_day in self.meta['recording_days'].values():
            for recording in recording_day['recordings']:
                self.objects["triangulation_recordings_objects"][recording].run_synchronization()
                for video in recording_day['recordings'][recording]["videos"]:
                    recording_day['recordings'][recording]["videos"][video]["marker_detection_filepath"] = str(self.objects["triangulation_recordings_objects"][recording].triangulation_dlc_cams_filepaths[video])
                    recording_day['recordings'][recording]["videos"][video]["synchronized_video"] = str(self.objects["triangulation_recordings_objects"][recording].synchronized_videos[video])
                    recording_day['recordings'][recording]["videos"][video]["framenum_synchronized"] = iio.v2.get_reader(self.objects["triangulation_recordings_objects"][recording].synchronized_videos[video]).count_frames()
            
    def _create_video_dict(self, video: VideoMetadata, intrinsics: bool=False)->Dict:
        dictionary = {'cam_id': video.cam_id, 
                    'filepath': str(video.filepath), 
                    'fps': video.fps, 
                    'framenum': iio.v2.get_reader(video.filepath).count_frames(), 
                    'exclusion_state': 'valid'}
        if intrinsics:
            dictionary['intrinsic_calibration_filepath'] = str(video.intrinsic_calibration_filepath)
        return dictionary
    
    def create_calibrations(self)->None:
        self.objects['calibration_objects'] = {}
        self.objects['position_objects'] = {}
        for recording_day in self.meta['recording_days'].values():
            for recording in recording_day['recordings']:
                videos = [video for video in recording_day['recordings'][recording]["videos"].keys() if recording_day['recordings'][recording]["videos"][video]['exclusion_state'] == 'valid']
                
                calibration_string = ""
                videos.sort()
                for elem in videos:
                    calibration_string = calibration_string + "_" + elem
                recording_day['recordings'][recording]['calibration_to_use'] = recording_day['recording_date'] + calibration_string
            
            calibration_object = Calibration(calibration_directory = recording_day['calibration_directory'], 
                project_config_filepath = self.project_config_path,
                recording_config_filepath = recording_day['recording_config_path'],
                output_directory = recording_day['calibration_directory'],
                overwrite = False)
            cams = [video.cam_id for video in calibration_object.metadata_from_videos]
            key = ""
            cams.sort()
            for elem in cams:
                key = key + "_" + elem
            key = recording_day['recording_date'] + key
            #possibility to create subgroups!!!
            self.objects['calibration_objects'][key] = calibration_object
            videos = {video.cam_id: self._create_video_dict(video, intrinsics=True) for video in calibration_object.metadata_from_videos}
            recording_day['calibrations'][key] = {
                        'target_fps': calibration_object.target_fps,
                        'led_pattern': calibration_object.led_pattern,
                        'videos': videos,
                        'key': key}
            #possibility to add only subgroup (=videos)!!!
            positions_object = Triangulation_Positions(positions_directory = recording_day['calibration_directory'], 
                    calibration_directory = recording_day['calibration_directory'],
                    recording_config_filepath = recording_day['recording_config_path'], 
                    project_config_filepath= self.project_config_path,
                    output_directory = recording_day['calibration_directory'], 
                    overwrite = False)
            self.objects['position_objects'][key] = positions_object
            for video in positions_object.metadata_from_videos:
                try:
                    recording_day['calibrations'][key]['videos'][video.cam_id]['positions_image_filepath'] = str(video.filepath) 
                except:
                    pass
                        
    def synchronize_calibrations(self)->None:
        for recording_day in self.meta['recording_days'].values():
            for calibration in recording_day['calibrations']:
                self.objects["calibration_objects"][calibration].run_synchronization()
                for video in recording_day['calibrations'][calibration]["videos"]:
                    recording_day['calibrations'][calibration]["videos"][video]["synchronized_video"] = str(self.objects["calibration_objects"][calibration].synchronized_charuco_videofiles[video])
                    recording_day['calibrations'][calibration]["videos"][video]["framenum_synchronized"] = iio.v2.get_reader(self.objects["calibration_objects"][calibration].synchronized_charuco_videofiles[video]).count_frames()

                self.objects["position_objects"][calibration].get_marker_predictions()
                for video in recording_day['calibrations'][calibration]["videos"]:
                    recording_day['calibrations'][calibration]["videos"][video]["positions_marker_detection_filepath"] = str(self.objects["position_objects"][calibration].triangulation_dlc_cams_filepaths[video])

              
    def calibrate(self, verbose: bool=False)->None:
        for recording_day in self.meta['recording_days'].values():
            for calibration in recording_day['calibrations']:
                self.objects["calibration_objects"][calibration].run_calibration(verbose=verbose)
                recording_day['calibrations'][calibration]['toml_filepath'] = str(self.objects["calibration_objects"][calibration].calibration_output_filepath)
                #add calibration log? reprojerr, etc.
        
        
    def triangulate_recordings(self)->None:
        for recording_day in self.meta['recording_days'].values():
            for recording in recording_day['recordings']:
                toml_path = recording_day['calibrations'][recording_day['recordings'][recording]["calibration_to_use"]]['toml_filepath']
                self.objects["triangulation_recordings_objects"][recording].run_triangulation(calibration_toml_filepath = toml_path)
                recording_day['recordings'][recording]['3D_csv'] = str(self.objects["triangulation_recordings_objects"][recording].csv_output_filepath)
                #add reprojection_error
