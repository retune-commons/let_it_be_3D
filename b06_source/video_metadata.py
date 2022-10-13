from typing import List, Tuple, Optional, Union, Dict
from abc import ABC, abstractmethod

from pathlib import Path
import datetime
import json
import pickle
import itertools as it
import imageio.v3 as iio
import numpy as np
import aniposelib as ap_lib
import cv2
import matplotlib.pyplot as plt

from b06_source.camera_intrinsics import IntrinsicCalibratorFisheyeCamera, IntrinsicCalibratorRegularCameraCharuco
from b06_source.utils import load_single_frame_of_video

class VideoMetadata:
    
    def __init__(self, filepath: Path, config_filepath: Path, max_frame_count: int = 300, load_calibration = False) -> None:
        if filepath.name.endswith('.mp4') or filepath.name.endswith('.AVI') or filepath.name.endswith('.jpg'):
            self.filepath = filepath
        else:
            raise ValueError ('The filepath is not linked to a video or image.')
        self._extract_filepath_metadata(filepath_name = filepath.name)
        if config_filepath.name.endswith('.json'):
            self._read_config_file(config_filepath=config_filepath)
        else:
            raise ValueError ('The config_filepath is not linked to a .json-file')
        self._get_intrinsic_parameters(config_filepath = config_filepath, max_frame_count = max_frame_count, load_calibration = load_calibration)
            
            
            
    @property
    def valid_cam_ids(self) -> List:
        return ['Ground1', 'Ground2', 'Side1', 'Side2', 'Side3', 'Top', 'Bottom']
    
    
    @property
    def valid_paradigms(self) -> List:
        return ['OTE', 'OTR', 'OTT']
    
    
    @property
    def valid_mouse_lines(self) -> List:
        return ['194', '195', '196', '206', '209', '210', '211']
    
    
    @property
    def intrinsic_calibrations_directory(self) -> Path:
        return Path('test_data/intrinsic_calibrations/')
    
        
    def _extract_filepath_metadata(self, filepath_name: str) -> None: 
        if filepath_name[-4:] == '.AVI':
            filepath_name = filepath_name.replace(filepath_name[filepath_name.index('00'):filepath_name.index('00')+3], '')
            self.cam_id = 'Top'
            
        if 'Charuco' in filepath_name or 'charuco' in filepath_name:
            self.charuco_video = True
            for piece in filepath_name[:-4].split('_'):
                if piece in self.valid_cam_ids:
                    self.cam_id = piece
                else: 
                    try:
                        self.recording_date = datetime.date(year = int('20' + piece[0:2]), month = int(piece[2:4]), day = int(piece[4:6]))
                    except ValueError:
                        pass
                
            for attribute in ['cam_id', 'recording_date']:
                if not hasattr(self, attribute):
                    self._check_attribute(attribute_to_check = attribute)
                    
        elif 'Positions' in filepath_name or 'positions' in filepath_name:
            for piece in filepath_name[:-4].split('_'):
                if piece in self.valid_cam_ids:
                    self.cam_id = piece
                else: 
                    try:
                        self.recording_date = datetime.date(year = int('20' + piece[0:2]), month = int(piece[2:4]), day = int(piece[4:6]))
                    except ValueError:
                        pass
                    
            for attribute in ['cam_id', 'recording_date']:
                if not hasattr(self, attribute):
                    self._check_attribute(attribute_to_check = attribute)
            
        else:
            self.charuco_video = False
            for piece in filepath_name[:-4].split('_'):
                if piece in self.valid_cam_ids:
                    self.cam_id = piece
                elif piece in self.valid_paradigms:
                    self.paradigm = piece
                elif piece in self.valid_mouse_lines:
                    self.mouse_line = piece
                elif piece.startswith('F'):
                    sub_pieces = piece.split('-')
                    if len(sub_pieces)==2:
                        try:
                            int(sub_pieces[1])
                            self.mouse_number = piece
                        except ValueError:
                            pass
                else: 
                    try:
                        self.recording_date = datetime.date(year = int('20' + piece[0:2]), month = int(piece[2:4]), day = int(piece[4:6]))
                    except ValueError:
                        pass
                
            for attribute in ['cam_id', 'recording_date', 'paradigm', 'mouse_line', 'mouse_number']:
                if not hasattr(self, attribute):
                    self._check_attribute(attribute_to_check = attribute)
            self.mouse_id = self.mouse_line + '_' + self.mouse_number
        
        
    def _check_attribute(self, attribute_to_check: str)->None:
        # TODO: input or set_defaults?
        print(f'{attribute_to_check} could not be extracted automatically for the following file:\n'
            f'{self.filepath}')
            
        messages = {}
        while True:
            entered_input = input(attribute_to_check + ': ')
        
            if attribute_to_check == 'cam_id':
                if entered_input in self.valid_cam_ids:
                    self.cam_id = entered_input
                    break
                else:
                    messages['cam_id'] = f'Entered cam_id {entered_input} did not match any of the defined cam_ids. \nPlease enter one of the following ids: {self.valid_cam_ids}'
            if attribute_to_check == 'recording_date':
                try:
                    self.recording_date = datetime.date(year = int('20' + entered_input[0:2]), month = int(entered_input[2:4]), day = int(entered_input[4:6]))
                    break
                except ValueError:
                    messages['recording_date'] = f'Entered recording_date {entered_input} does not match the required structure for date. \nPlease enter the date as YYMMDD , e.g., 220928.'
            if attribute_to_check == 'paradigm':
                if entered_input in self.valid_paradigms:
                    self.paradigm = entered_input
                    break
                else:
                    messages['paradigm'] = f'Entered paradigm does not match any of the defined paradigms. \nPlease enter one of the following paradigms: {self.valid_paradigms}'
            if attribute_to_check == 'mouse_line':
                if entered_input in self.valid_mouse_lines:
                    self.mouse_line = entered_input
                    break
                else:
                    messages['mouse_line'] = f'Entered mouse_line is not supported. \nPlease enter one of the following lines: {self.valid_mouse_lines}'
            if attribute_to_check == 'mouse_number':
                sub_pieces = entered_input.split('-')
                if len(sub_pieces)==2 and entered_input.startswith('F'):
                    try:
                        int(sub_pieces[1])
                        self.mouse_number = entered_input
                        break
                    except ValueError:
                        messages['mouse_number'] = f'Entered mouse_number does not match the required structure for a mouse_number. \n Please enter the mouse number as Generation-Number, e.g., F12-45'
                else:
                    messages['mouse_number'] = f'Entered mouse_number does not match the required structure for a mouse_number. \n Please enter the mouse number as Generation-Number, e.g., F12-45'
            print(messages[attribute_to_check])
            

    def _read_config_file(self, config_filepath: Path)->None:
        with open(config_filepath,'r') as jsonfile:
            config = json.load(jsonfile)
        for key in ['led_pattern', self.cam_id]:
            try:
                config[key]
            except KeyError:
                raise KeyError(f'Missing information for {key} in the config_file {config_filepath}!')
        self.led_pattern = config['led_pattern']
        metadata_dict = config[self.cam_id]
        
        for key in ['fps', 'offset_row_idx', 'offset_col_idx', 'flip_h', 'flip_v', 'fisheye']:
            try:
                metadata_dict[key]
            except KeyError:
                raise KeyError(f'Missing metadata information in the config_file {config_filepath} for {self.cam_id} for {key}.')      
        self.fps = metadata_dict['fps']
        self.offset_row_idx = metadata_dict['offset_row_idx']
        self.offset_col_idx = metadata_dict['offset_col_idx']
        self.flip_h = metadata_dict['flip_h']
        self.flip_v = metadata_dict['flip_v']
        self.fisheye = metadata_dict['fisheye']

        
    def _get_intrinsic_parameters(self, config_filepath: Path, load_calibration: bool, max_frame_count: int) -> None:
        if self.charuco_video:
            if self.fisheye:
                try:
                    intrinsic_calibration_filepath = self.intrinsic_calibrations_directory.joinpath('Bottom_checkerboard_intrinsic_calibration_results.p')
                    with open(intrinsic_calibration_filepath, 'rb') as io:
                        intrinsic_calibration = pickle.load(io)
                except FileNotFoundError:
                    try:
                        intrinsic_calibration_checkerboard_video_filepath = [file for file in self.intrinsic_calibrations_directory.iterdir() if file.name.endswith('.mp4') and 'checkerboard' in file.name and self.cam_id in file.name][0]
                        calibrator = IntrinsicCalibratorFisheyeCamera(filepath_calibration_video = intrinsic_calibration_checkerboard_video_filepath, max_frame_count = max_frame_count)
                        intrinsic_calibration = calibrator.run()
                    except IndexError:
                        raise FileNotFoundError (f'Could not find a filepath for an intrinsic calibration or a checkerboard video for {self.cam_id}.\nIt is required having a intrinsic_calibration .p file or a checkerboard video in the intrinsic_calibrations_directory ({self.intrinsic_calibrations_directory}) for a fisheye-camera!')
            else:
                if load_calibration:
                    try:
                        intrinsic_calibration_filepath = [file for file in self.intrinsic_calibrations_directory.iterdir() if file.name.endswith('.p') and self.cam_id in file.name][0]
                        with open(intrinsic_calibration_filepath, 'rb') as io:
                            intrinsic_calibration = pickle.load(io)
                    except IndexError:
                        raise FileNotFoundError (f'Could not find an intrinsic calibration for {self.cam_id}! Use "load_calibration = False" to calibrate now!')
                else:
                    calibrator = IntrinsicCalibratorRegularCameraCharuco(filepath_calibration_video = self.filepath, max_frame_count = max_frame_count)
                    intrinsic_calibration = calibrator.run()
            self._save_calibration(intrinsic_calibration = intrinsic_calibration)
        else:
            try:
                intrinsic_calibration_filepath = [file for file in config_filepath.parent.iterdir() if file.endswith('.p') and self.cam_id in file][0]
                with open(intrinsic_calibration_filepath, 'rb') as io:
                            intrinsic_calibration = pickle.load(io)
            except IndexError:
                raise FileNotFoundError (f'Could not find an intrinsic calibration for {self.cam_id} in {config_filepath.parent}! \nRunning the 3D Calibration should also create an intrinsic calibration. \nMake sure, you run the 3D Calibration before Triangulation.')
                
        adjusting_required = self._is_adjusting_of_intrinsic_calibration_required(unadjusted_intrinsic_calibration = intrinsic_calibration)
        self._set_intrinsic_calibration(intrinsic_calibration = intrinsic_calibration, adjusting_required = adjusting_required)
        
        # ToDo:
        # We might need some methods to inspect the quality of the intrinsic calibration
        # after adjusting it to the cropped video, since flipping the video streams in
        # ICcapture has different effects, depending on whether it was applied right from
        # launching the software / cameras, or whether it was manually activated once the
        # software is running and the camera was loaded. This is at least our best guess
        # for the inconsistent behavior & warrants further testing.
        
    def _save_calibration(self, intrinsic_calibration: Dict) -> None:
        video_filename = self.filepath.name
        filename = f'{video_filename[:video_filename.rfind(".")]}_intrinsic_calibration_results.p'
        with open(self.filepath.parent.joinpath(filename), 'wb') as io:
            pickle.dump(intrinsic_calibration, io)
        
    def _is_adjusting_of_intrinsic_calibration_required(self, unadjusted_intrinsic_calibration: Dict) -> bool:
        adjusting_required = False
        if any([self.offset_col_idx != 0, self.offset_row_idx != 0, self.flip_h, self.flip_v]):
            adjusting_required = True
        return adjusting_required     
        

    def _set_intrinsic_calibration(self, intrinsic_calibration: Dict, adjusting_required: bool) -> None:
        if adjusting_required:
            intrinsic_calibration = self._adjust_intrinsic_calibration(unadjusted_intrinsic_calibration = intrinsic_calibration)
        setattr(self, 'intrinsic_calibration', intrinsic_calibration)
        
        
    def _adjust_intrinsic_calibration(self, unadjusted_intrinsic_calibration: Dict) -> Dict:
        adjusted_intrinsic_calibration = unadjusted_intrinsic_calibration.copy()
        intrinsic_calibration_video_size = unadjusted_intrinsic_calibration['size']
        new_video_size = self._get_cropped_video_size()
        self._get_correct_x_y_offsets(intrinsic_calibration_video_size = intrinsic_calibration_video_size, new_video_size = new_video_size)
        adjusted_K = self._get_adjusted_K(K = unadjusted_intrinsic_calibration['K'])
        adjusted_intrinsic_calibration = self._incorporate_adjustments_in_intrinsic_calibration(intrinsic_calibration = unadjusted_intrinsic_calibration.copy(),
                                                                                                new_size = new_video_size,
                                                                                                adjusted_K = adjusted_K)
        return adjusted_intrinsic_calibration
    
    
    def _get_cropped_video_size(self) -> Tuple[int, int]:
        props = iio.improps(self.filepath, index = 0)
        size = props.shape[0:2]
        return size
    
    
    def _get_correct_x_y_offsets(self, intrinsic_calibration_video_size: Tuple[int, int], new_video_size: Tuple[int, int]) -> None:
        if self.flip_v:
            self.offset_row_idx = intrinsic_calibration_video_size[0] - new_video_size[1] - self.offset_row_idx
            # rows or cols first in intrinsic_calibration_video_size? (rows for now, but maybe this will be changed?)
        if self.flip_h:
            self.offset_col_idx = intrinsic_calibration_video_size[1] - new_video_size[0] - self.offset_col_idx
    
    
    def _get_adjusted_K(self, K: np.ndarray) -> np.ndarray:
        adjusted_K = K.copy()
        adjusted_K[0][2] = adjusted_K[0][2] - self.offset_row_idx
        adjusted_K[1][2] = adjusted_K[1][2] - self.offset_col_idx
        return adjusted_K  
    
    
    def _incorporate_adjustments_in_intrinsic_calibration(self, intrinsic_calibration: Dict, new_size: Tuple[int, int], adjusted_K: np.ndarray) -> Dict:
        intrinsic_calibration['size'] = new_size
        intrinsic_calibration['K'] = adjusted_K
        return intrinsic_calibration
    
    
    