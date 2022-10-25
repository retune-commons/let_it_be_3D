<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
# This Repository is still work in progress
We are doing our best to enable people to use this 2D to 3D pipeline as soon as possible!

# let_it_be_3D

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

<style type="text/css">
  td {
    padding: 0 15px;
  }
</style>

</tr>
</table>


With let_it_be_3D we try to simplify 2D to 3D conversion of animal tracking data.

## Pipeline
### 1) Preprocess data to meet Anipose input requirements
- Remove distortion by calibrating single-cameras intrinsically
- Synchronize video recordings of different frame rates in time

### 2) Markerdetection

- For anipose calibration, recording analysis and ground truth estimation
- Using either DeepLabCut, Template matching or manual annotations

### 3) Create 3D-reconstruction
Using Anipose, we create a 3D-reconstruction of both our recording, as well as our ground truth data.

### 4) Quality assurance
- 3D camera calibration (compare real-world & triangulation space)
- Ground truth validation

## Contributers
The pipeline was designed by [Dennis Segebarth](https://github.com/DSegebarth), [Konstantin Kobel](https://github.com/KonKob) and [Michael Schellenberger](https://github.com/MSchellenberger).
At the Sfb-Retune Hackathon 2022, [Elisa Garulli](https://github.com/ELGarulli), [Robert Peach](https://github.com/peach-lucien) and [Veronika Selzam](https://github.com/vselzam)
joined the taskforce to push the project towards completion. 

If you want to help with writing this pipeline, please <a href = "mailto: schellenb_m1@ukw.de">get in touch</a>.


