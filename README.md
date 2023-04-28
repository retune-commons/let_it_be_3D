# let_it_be_3D

With let_it_be_3D we want to extend the functions of aniposelib and bring them into a pipeline structure.
Our goals are having as less manual steps required as possible and standardized quality assurance/collection of metadata.
We provide additional methods for `video synchronisation`, `adjustment for different framerates`, `validation of anipose calibration`, `adjustment of intrinsic calibrations to croppings`, `manual marker detection`, `checking and correcting filenames` and `normalisation of the 3D triangulated dataframe`.

<details>
<summary> See the pipeline flowchart! </summary>

```mermaid
flowchart TD;
    video_dir_R(Recording directory) ~~~ video_dir_C(Calibration directory);
    id1(Recording object) ~~~ id2(Calibration object) ~~~ id3(Calibration validation objects);
    subgraph Processing recording videos:
    video_dir_R --> |Get video metadata \nfrom filename and recording config| id1;
    id1-->|Temporal synchronisation| id4>DeepLabCut analysis and downsampling];
    end
    subgraph Processing calibration videos
    video_dir_C --> |Get video metadata \nfrom filename and recording config| id2 & id3;
    id2-->|Temporal synchronisation| id5>Video downsampling];
    id3-->id6>Marker detection];
    end
    id5-->id7{Anipose calibration};
    subgraph Calibration validation
    id7-->id8[/Good calibration reached?/];
    id6-->id8;
    end
    subgraph Triangulation
    id8-->|No|id7;
    id8-->|Yes|id9>Triangulation];
    id4-->id9-->id10>Normalization];
    id10-->id11[(Database)];
    end
```

</details>

<details>
<summary> Pipeline explained Step-by-Step! </summary>


### 1) Load videos and metadata
- read video metadata from filename and recording config file
- intrinsic calibrations
    - use anipose intrinsic calibration
    - run or load intrinsic calibration based on uncropped checkerboard videos
    adjust intrinsic calibration for video cropping
      <details>
      <summary>Example </summary>
        ToDo:(undistorted image???) 
      </details>
### 2) Video processing
- synchronize videos temporally based on a blinking signal 
    <details>
    <summary>Example </summary>
        <p align="left">
        <img src="https://user-images.githubusercontent.com/104254966/234807715-fede2f67-e6b0-4eef-81aa-13c16a5ffb79.png" width="350">
        </p>
    </details>
- run marker detection on videos manually or using [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) networks 
  <details>
  <summary>Example </summary>
      <p align="left">
      <img src="https://user-images.githubusercontent.com/104254966/234822304-f19d62d3-9fed-410a-8267-abd8fd43d24a.png" width="350">
      <img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1628250004229-KVYD7JJVHYEFDJ32L9VJ/DLClogo2021.jpg?format=1000w" width="30%">
      </p>
  </details>
- write videos and marker detection files to the same framerate

### 3) Calibration
- run extrinsic [Anipose](https://github.com/lambdaloop/aniposelib) camera calibration 
- validate calibration based on known distances and angles (ground truth) between calibration validation markers
  <details>
  <summary>Example </summary>
        This calibration validation shows the triangulated representation of a tracked rectangle, that has 90° angles at the corners.
        <p align="left">
        <img src="https://user-images.githubusercontent.com/104254966/234811649-5e22dc44-99d9-410f-9db3-191603151b4d.png" width="350">
        </p>
  </details>

### 4) Triangulation
- triangulate recordings
  <details>
  <summary>Example </summary>
    <p align="left">
    <img src="https://user-images.githubusercontent.com/104254966/234822258-5ad2815b-362e-4370-a257-4d925c14ab13.png" width="350">
    </p>
  </details>
  
- rotate dataframe, translate to origin, normalize to centimeter

  <details>
  <summary>Example </summary>
    The blue vectors were aligned to the yellow vectors succesfully.
        <p align="left">
        <img src="https://user-images.githubusercontent.com/104254966/234811752-b6c5b5af-ab71-4c10-8099-1dbd93d8c3f0.png" width="350" title="hover text">
        </p>
  </details>
    
- add metadata to database
</details>


## How to use
### Installation
```bash
# Clone this repository
$ git clone https://github.com/retune-commons/let_it_be_3D.git

# Go to the folder in which you cloned the repository
$ cd let_it_be_3D

# Install dependencies
# first, install deeplabcut into a new environment as described here: (https://deeplabcut.github.io/DeepLabCut/docs/installation.html)
$ conda env update --file env.yml 

# Open Walkthrough.ipynb in jupyter lab
$ jupyter lab

# Update project_config.yaml to your needs and you're good to go!
```


### Examples
<details>
<summary>Calibration</summary>

  ```ruby
  from pathlib import Path
  from core.triangulation_calibration_module import Calibration
  rec_config = Path("test_data/Server_structure/Calibrations/220922/recording_config_220922.yaml")
  calibration_object = Calibration(
    calibration_directory=rec_config.parent,
    recording_config_filepath=rec_config,
    project_config_filepath="test_data/project_config.yaml",
    output_directory=rec_config.parent,
  )
  calibration_object.run_synchronization()
  calibration_object.run_calibration(verbose=2)
  ```
  
</details>
<details>
<summary>TriangulationRecordings </summary>

  ```ruby
    from core.triangulation_calibration_module import TriangulationRecordings
    rec_config = "test_data/Server_structure/Calibrations/220922/recording_config_220922.yaml"
    directory = "test_data/Server_structure/VGlut2-flp/September2022/206_F2-63/220922_OTE/"
    triangulation_object = TriangulationRecordings(
      directory=directory,
      recording_config_filepath=rec_config,
      project_config_filepath="test_data/project_config.yaml",
      recreate_undistorted_plots = True,
      output_directory=directory
    )
    triangulation_object.run_synchronization()
    triangulation_object.exclude_markers(
      all_markers_to_exclude_config_path="test_data/markers_to_exclude_config.yaml",
      verbose=False,
    )
    triangulation_object.run_triangulation(
      calibration_toml_filepath="test_data/Server_structure/Calibrations/220922/220922_0_Bottom_Ground1_Ground2_Side1_Side2_Side3.toml"
    )
    normalised_path, normalisation_error = triangulation_object.normalize(
      normalization_config_path="test_data/normalization_config.yaml"
    )
  ```
  
</details>
<details>
<summary>CalibrationValidation </summary>

  ```ruby
    from core.triangulation_calibration_module import CalibrationValidation
    from pathlib import Path
    rec_config = Path("test_data/Server_structure/Calibrations/220922/recording_config_220922.yaml")
    calibration_validation_object = CalibrationValidation(
      project_config_filepath="test_data/project_config.yaml",
      directory=rec_config.parent, recording_config_filepath=rec_config,
      recreate_undistorted_plots = True, output_directory=rec_config.parent
    )
    calibration_validation_object.add_ground_truth_config("test_data/ground_truth_config.yaml")
    calibration_validation_object.get_marker_predictions()
    calibration_validation_object.run_triangulation(
      calibration_toml_filepath="test_data/Server_structure/Calibrations/220922/220922_0_Bottom_Ground1_Ground2_Side1_Side2_Side3.toml",
      triangulate_full_recording = True
    )
    mean_dist_err_percentage, mean_angle_err, reprojerr_nonan_mean = calibration_validation_object.evaluate_triangulation_of_calibration_validation_markers()
  ```
  
</details>

### Required filestructure
<details>
<summary>Video filename </summary>

  - calibration:
    - has to be a `video` [".AVI", ".avi", ".mov", ".mp4"]
    - including recording_date (YYMMDD), calibration_tag (as defined
    in project_config) and cam_id (element of valid_cam_ids in
    project_config)
    - recording_date and calibration_tag have to be separated by an
    underscore ("_")
    - f"{recording_date}_{calibration_tag}_{cam_id}" =
    Example: "220922_charuco_Front.mp4"
  - calibration_validation:
    - has to be a `video` or `image` [".bmp", ".tiff", ".png", ".jpg",
    ".AVI", ".avi", ".mp4"]
    - including recording_date (YYMMDD), calibration_validation_tag
    (as defined in project_config) and cam_id (element of valid_cam_ids
    in project_config)
    - recording_date and calibration_validation_tag have to be separated
    by an underscore ("_")
    - calibration_validation_tag mustn't be "calvin"
    - f"{recording_date}_{calibration_validation_tag}" =
    Example: "220922_position_Top.jpg"
  - recording:
    - has to be a `video` [".AVI", ".avi", ".mov", ".mp4"]
    - including recording_date (YYMMDD),
    cam_id (element of valid_cam_ids in project_config),
    mouse_line (element of animal_lines in project_config),
    animal_id (beginning with F, split by "-" and followed by a number)
    and paradigm (element of paradigms in project_config)
    - recording_date, cam_id, mouse_line, animal_id and paradigm have to be separated by an underscore ("_")
    - f"{recording_date}_{cam_id}_{mouse_line}_{animal_id}_{paradigm}.mp4" =
    Example: "220922_Side_206_F2-12_OTT.mp4"
    
</details>

<details>
<summary>Folder structure </summary>

  - A folder, in which a recordings is stored should match the followed structure to be 
    detected automatically:
    - has to start with the recording_date (YYMMDD)
    - has to end with any of the paradigms (as defined in project_config)
    - recording date and paradigm have to be separated by an underscore ("_")
    - f"{recording_date}_{paradigm}" = 
    Example: "230427_OF"
    
</details>

### API Documentation
Please see our API-documentation [here](https://let-it-be-3d.readthedocs.io/en/latest/)!

## License
GNU General Public License v3.0

## Contributers
The pipeline was designed by [Konstantin Kobel](https://github.com/KonKob), [Dennis Segebarth](https://github.com/DSegebarth) and [Michael Schellenberger](https://github.com/MSchellenberger).
At the Sfb-Retune Hackathon 2022, [Elisa Garulli](https://github.com/ELGarulli), [Robert Peach](https://github.com/peach-lucien) and [Veronika Selzam](https://github.com/vselzam)
joined the taskforce to push the project towards completion. 

<table>
<tr>
<td>
    <a href="https://sfb-retune.de/"> 
        <img src="https://sfb-retune.de/images/logo-retune.svg" alt="Sfb-Retune" style="width: 250px;"/>
    </a>
</td> 
<td>
    <a href="https://www.defense-circuits-lab.com/"> 
        <img src="https://static.wixstatic.com/media/547baf_87ffe507a5004e29925dbeb65fe110bb~mv2.png/v1/fill/w_406,h_246,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/LabLogo3black.png" alt="DefenseCircuitsLab" style="width: 250px;"/>
    </a>
</td>
</tr>
</table>

## Contact
If you want to help with writing this pipeline, please <a href = "mailto: schellenb_m1@ukw.de">get in touch</a>.
