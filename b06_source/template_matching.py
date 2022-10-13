from typing import List, Tuple, Dict, Optional
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import match_template
from skimage.io import imsave
import imageio.v3 as iio



class CreateNewTemplates:
    
    def __init__(self, image_filepath: Path, cam_id: str) -> None:
        self.image_filepath = image_filepath
        self.cam_id = cam_id
    
    def create_templates_for_marker_id(self, marker_id: str, 
                                       marker_id_coords: Tuple[int, int], 
                                       offsets_to_upper_left_corner: Tuple[int, int] = (25, 25),
                                       template_shape: Tuple[int, int] = (100, 100),
                                       template_level_depth: int = 3,
                                       zoom_factor_per_level: float = 0.25,
                                       save_templates: bool=True,
                                       show_plot: bool=True) -> None:
        depth_level = 0
        image_corner_coords_per_level = []
        base_output_filename = self._get_basis_for_output_filename(marker_id = marker_id)
        upper_left_corner_coords = self._get_upper_left_from_marker_coords(marker_coords = marker_id_coords, 
                                                                         offsets_to_upper_left_corner = offsets_to_upper_left_corner)
        image_corner_coords = self._get_image_corner_coords(upper_left_corner_coords = upper_left_corner_coords, template_shape = template_shape)
        full_output_filename = self._extend_base_output_filename(base_filepath = base_output_filename,
                                                                 offsets_to_upper_left_corner = offsets_to_upper_left_corner,
                                                                 depth_level = depth_level)
        image_corner_coords_per_level.append(image_corner_coords)
        if save_templates:
            self._crop_image_arround_marker_and_save_template_image(output_filepath = full_output_filename, image_corner_coords = image_corner_coords)
        for i in range(template_level_depth - 1):
            depth_level += 1
            offsets_to_upper_left_corner, template_shape = self._update_for_next_depth_level(offsets_to_upper_left_corner = offsets_to_upper_left_corner,
                                                                                             template_shape = template_shape,
                                                                                             zoom_factor_per_level = zoom_factor_per_level)
            upper_left_corner_coor = self._get_upper_left_from_marker_coords(marker_coords = marker_id_coords, 
                                                                             offsets_to_upper_left_corner = offsets_to_upper_left_corner)
            image_corner_coords = self._get_image_corner_coords(upper_left_corner_coords = upper_left_corner_coords, template_shape = template_shape)
            full_output_filename = self._extend_base_output_filename(base_filepath = base_output_filename,
                                                                     offsets_to_upper_left_corner = offsets_to_upper_left_corner,
                                                                     depth_level = depth_level)
            image_corner_coords_per_level.append(image_corner_coords)
            if save_templates:
                self._crop_image_arround_marker_and_save_template_image(output_filepath = full_output_filename, image_corner_coords = image_corner_coords)
        if show_plot:
            self._plot_all_template_levels_on_positions_image(image_corner_coords_per_level = image_corner_coords_per_level)
            
            
    def _get_basis_for_output_filename(self, marker_id: str) -> Path:
        return self.image_filepath.parent.joinpath(f'{self.cam_id}_template_{marker_id}')


    def _get_upper_left_from_marker_coords(self, marker_coords: Tuple[int, int], offsets_to_upper_left_corner: Tuple[int, int]) -> Tuple[int, int]:
        upper_left_row_idx = marker_coords[0] - offsets_to_upper_left_corner[0]
        upper_left_col_idx = marker_coords[1] - offsets_to_upper_left_corner[1]
        return upper_left_row_idx, upper_left_col_idx


    def _get_image_corner_coords(self, upper_left_corner_coords: Tuple[int, int], template_shape: Tuple[int, int]) -> Dict:
        upper_right_coords = (upper_left_corner_coords[0], upper_left_corner_coords[1] + template_shape[1])
        lower_left_coords = (upper_left_corner_coords[0] + template_shape[0], upper_left_corner_coords[1])
        lower_right_coords = (upper_left_corner_coords[0] + template_shape[0], upper_left_corner_coords[1] + template_shape[1])
        all_corner_coords = {'upper_left': upper_left_corner_coords,
                             'upper_right': upper_right_coords,
                             'lower_left': lower_left_coords,
                             'lower_right': lower_right_coords}
        return all_corner_coords
        
        
    def _extend_base_output_filename(self, base_filepath: Path, offsets_to_upper_left_corner: Tuple[int, int], depth_level: int) -> Path:
        marker_id = base_filepath.name[base_filepath.name.find('template_') + 9:]
        templates_already_present = []
        for file in base_filepath.parent.iterdir():
            if (self.cam_id in file.name) and ('template' in file.name) and (marker_id in file.name) and (f'lvl-{depth_level}' in file.name):
                templates_already_present.append(file.name)
        template_idx = str(len(templates_already_present)).zfill(2)
        return self.image_filepath.parent.joinpath(f'{base_filepath.name}_offset-{offsets_to_upper_left_corner[0]}-{offsets_to_upper_left_corner[1]}_lvl-{depth_level}_{template_idx}.png')


    def _crop_image_arround_marker_and_save_template_image(self, output_filepath: Path, image_corner_coords: Dict) -> None:
        image = iio.imread(self.image_filepath)
        row_cropping_idxs = (image_corner_coords['upper_left'][0], image_corner_coords['lower_left'][0])
        column_cropping_idxs = (image_corner_coords['upper_left'][1], image_corner_coords['upper_right'][1])
        template = image[row_cropping_idxs[0] : row_cropping_idxs[1], column_cropping_idxs[0] : column_cropping_idxs[1]].copy()
        imsave(output_filepath, template)


    def _update_for_next_depth_level(self, offsets_to_upper_left_corner: Tuple[int, int], 
                                     template_shape: Tuple[int, int], 
                                     zoom_factor_per_level: float) -> Tuple[Tuple[int, int], Tuple[int, int]]:   
        new_offsets_to_upper_left_corner = tuple([int(coordinate * zoom_factor_per_level) for coordinate in offsets_to_upper_left_corner])
        new_template_shape = tuple([int(n_pixels * zoom_factor_per_level) for n_pixels in template_shape])
        return new_offsets_to_upper_left_corner, new_template_shape


    def _plot_all_template_levels_on_positions_image(self, image_corner_coords_per_level: List[Dict]) -> None:
        #fig = plt.figure(figsize=(15, 10))
        plt.imshow(iio.imread(self.image_filepath))
        for image_corner_coords in image_corner_coords_per_level:
            rows, cols = [], []
            for corner_id in ['upper_left', 'upper_right', 'lower_right', 'lower_left', 'upper_left']:
                rows.append(image_corner_coords[corner_id][0])
                cols.append(image_corner_coords[corner_id][1])
            print(rows, cols)
            plt.plot(cols, rows)
        plt.show()