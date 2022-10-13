"""
To do:
    
    GUI instead of input
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
    
    Input: none
    UI-input: target_fps, intristic calibration_dir, animal_lines, paradigms, camID processing_type, processing_path, calibration_evaluation_type, calibration_evaluation_path
    Output: saves let_it_be_3D_project_config in cwd (includes all UI-inputs and list of cams -> valid_cam_IDs)
    """
    
    
    
    def get_user_input(self):
        
        user_input = {'processing_type': {}, 'processing_path': {}, 'calibration_evaluation_type': {}, 'calibration_evaluation_path': {}}
        user_input['target_fps'] = int(input('targetfps:'))
        user_input['valid_cam_IDs'] = []
        user_input['intrinsic_calibration_dir'] = input('intrinsic_calibration_dir:')
        animal_lines = input('Please write the animal lines as a list of numbers seperated by commas ( e.g. "195,206,209" )')
        user_input['animal_lines'] = [elem.strip() for elem in animal_lines.split(',')]
        paradigms = input('Please write the paradigm types lines as a list ( e.g. ""OTR", "OTT", "OTE"" )')
        user_input['paradigms'] = [elem.strip() for elem in paradigms.split(',')]
        adding_cams_ongoing = True
        while adding_cams_ongoing == True:
            
            camID = input('camID: ')          
            user_input['processing_type'][camID] =  input('what´s the processing_type? (DLC/TM/Manual)')      
            user_input['processing_path'][camID] = input('what´s the processing path?')
            user_input['calibration_evaluation_type'][camID] =  input('what´s the calibration evaluation type (DLC/TM/Manual?')        
            user_input['calibration_evaluation_path'][camID] = input('what´s the calibration_evaluation path?')
            
            user_input['valid_cam_IDs'].append(camID)
            
            if input('Add more cams?') == 'yes':
                continue
            
            else:
                if input('Are you sure? (if so print: yes)'):    
                    adding_cams_ongoing = False
                else: continue
                    
        self.user_input = user_input
            
    def write_config(self):
        with open('.yaml', 'w') as write:
            yaml.dump(self.user_input, write)
    
    
class RecordingConfigs(Configs):
    """Configs subclass for information about the individual recording day
    input: path of the project config file, which is named
    UI-input: recording_date, led_pattern, fps, offset_row_idx, offset_col_idx, fliph, flipv, fisheye
    output
    
    """
    
    def load_projectconfig(self, path_project_config = None):
        with open(path_project_config, 'r') as file:
            self.project_config = yaml.safe_load(file)
        
    def user_input(self):
        user_input = {}
        user_input['recording_date'] = input('Please write the recording date in the yymmdd format')
        user_input['led_pattern'] = input('What´s the LED pattern?')
        
        print('The following cameras have been found:', self.project_config['valid_cam_IDs'])
        for cam in (self.project_config['valid_cam_IDs']):
            print('\n--------------------------------------------------\n \
            currently working on cameraID: ' + str(cam) + '\
                 \n--------------------------------------------------\n')
            user_input[cam] = {}
            user_input[cam]['fps'] = int(input('fps'))
            user_input[cam]['offset_row_idx'] = int(input('offset row id x'))
            user_input[cam]['offset_col_idx'] = int(input('offset_col_idx'))
            user_input[cam]['fliph'] = input('flip horizontal write "True" for True  and "False" for False')
            user_input[cam]['flipv'] = input('flip vertical write "True" for True  and "False" for False')
            user_input[cam]['fisheye'] = input('fisheye write "True" for True  and "False" for False')
        
        self.user_input = user_input
        
    def write_config(self):
        recording_name = 'let_it_be_3D_recording_config_' + self.user_input['recording_date'] + '.yaml'
        with open(recording_name, 'w') as write:
            print('\nRecording config successfully written!')
            yaml.dump(self.user_input, write)