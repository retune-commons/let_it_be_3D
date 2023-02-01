
import os
import numpy as np
import pandas as pd

from b06_source.camera_calibration import CalibrationForAnipose3DTracking, SingleCamDataForAnipose

def find_optimal_calibration(single_cams,
                             config, 
                             precomputed=False,
                             path_to_cal='./',
                             max_iters=10,
                             p_threshold=0.1,
                             angle_threshold=5,
                             ):
    
    """ finds optimal calibration either from a folder or through repeated optimisations of anipose """
    
    if precomputed:
        calibration_files = os.listdir(path_to_cal)
        max_iters = len(calibration_files)
    
    report = pd.DataFrame()
    
    calibration_found = False 
    
    for cal in range(max_iters):
                    
        anipose_calibration = CalibrationForAnipose3DTracking(single_cams_to_calibrate = single_cams)
        
        if not precomputed:
            anipose_calibration.run_calibration()
            calibration_file = path_to_cal +'calibration_{}.toml'.format(cal)
            anipose_calibration.save_calibration(calibration_file)
            
        else:            
            calibration_file  = os.path.join(path_to_cal, calibration_files[cal])
            anipose_calibration.load_calibration(filepath = calibration_file)
            
        
        anipose_calibration.evaluate_triangulation_of_test_position_markers(config)
        calibration_errors = anipose_calibration.anipose_io['distance_errors_in_cm']
        calibration_angles_errors = anipose_calibration.anipose_io['angles_error_screws_plan']

        for reference in calibration_errors.keys():
            all_percentage_errors = [percentage_error for marker_id_a, marker_id_b, distance_error, percentage_error in calibration_errors[reference]['individual_errors']]
        
        for reference in calibration_angles_errors.keys():
            all_angle_errors = list(calibration_angles_errors.values())

        mean_dist_err_percentage = np.asarray(all_percentage_errors).mean()
        mean_angle_err = np.asarray(all_angle_errors).mean()

        print("Calibration {}".format(calibration_file) +
              "\n mean percentage error: "+ str(mean_dist_err_percentage) + 
              "\n mean angle error: "+ str(mean_angle_err) )        

        report.loc[calibration_file, 'mean_distance_error_percentage'] = mean_dist_err_percentage
        report.loc[calibration_file, 'mean_angle_error'] = mean_angle_err
        
        
        if mean_dist_err_percentage < p_threshold and mean_angle_err < angle_threshold:
            calibration_found = True
            print("Good Calibration reached: \n" +
                 "Calibration {}".format(calibration_file) +
                      "\n mean percentage error: "+ str(mean_dist_err_percentage) + 
                      "\n mean angle error: "+ str(mean_angle_err))
            
            break
        
    
    if not calibration_found:
        print('No optimal calibration found with given thresholds')

    
    return report
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    