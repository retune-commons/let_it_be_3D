# let_it_be_3D

With let_it_be_3D we want to extend the functions of aniposelib and bring them into a pipeline structure.
As less manual steps required as possible, standardized quality assurance and collection of metadata.
- video synchronisation
- framerate adjustment
- calibration validation
- manual marker detection
- filename checker
- normalisation

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
    id8-->|Yes|id9(Triangulation);
    id4-->id9-->id10(Normalization);
    id10-->id11[(Database)];
    end
```

</details>

<details>
<summary> Pipeline explained Step-by-Step! </summary>


### 1) Load videos and metadata
- read video metadata from [filename](#filename_structure) and recording config file
- intrinsic calibrations
    - use anipose intrinsic calibration
    - run or load intrinsic calibration based on uncropped checkerboard videos (undistorted image???) 
    adjust intrinsic calibration for video cropping

### 2) Video processing
- synchronize videos temporally based on a blinking signal 
    <details>
    <summary>Example </summary>
        <p align="center">
        <img src="https://user-images.githubusercontent.com/104254966/234807715-fede2f67-e6b0-4eef-81aa-13c16a5ffb79.png" width="350" title="hover text">
        </p>
    </details>
- run marker detection on videos manually or using DeepLabCut networks (DLC label), (2D dataframes)
- write videos and marker detection files to the same framerate

### 3) Calibration
- run extrinsic Anipose camera calibration (anipose label)
- validate calibration based on known distances and angles (ground truth) between calibration validation markers (calvin image!)
    <details>
    <summary>Example </summary>
        This calibration validation shows the triangulated representation of a tracked rectangle, that has 90° angles at the corners.
        <p align="center">
        <img src="https://user-images.githubusercontent.com/104254966/234811649-5e22dc44-99d9-410f-9db3-191603151b4d.png" width="350" title="hover text">
        </p>
    </details>

### 4) Triangulation
- triangulate recordings (3D dataframes)
- rotate dataframe, translate to origin, normalize to centimeter
    <details>
    <summary>Example </summary>
    
        <p align="center">
        <img src="https://user-images.githubusercontent.com/104254966/234811752-b6c5b5af-ab71-4c10-8099-1dbd93d8c3f0.png" width="350" title="hover text">
        </p>

    </details>
    
- add metadata to database

</details>



## How to use
### Installation

### Examples

### Required filestructure
#### <a name="filename_structure"></a>Video filename

#### Folder structure


### API Documentation
Please see our API-documentation [here](https://let-it-be-3d.readthedocs.io/en/latest/)!

## Why 3D?

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
