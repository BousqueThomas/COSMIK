# <ins>COSMIK</ins>

Welcome on the COSMIK repository.
This repo gives you a pipeline that permits you to have Inverse Kinematics thanks to basic videos obtained with as minimum two cameras.

You will find here the result of my work carried out during my Master 2 internship.

<br />

## Processing

You have two possibilities :
1) Launch each step of the pipe one by one
2) Launch all the pipeline directly

<br />
    
**Option 1 : step by step**

You will be able to find each code in the folder **apps**.

- ```mmpose_processing.py``` : to have JCPs (Joint Center Positions) from a video. <ins>You need to process this code for each cameras and each video.</ins>
- ```fusion_models.py``` : to concatenate the results of two mmpose models (body26 and wholebody). The goal is to add hands.
- ```triangulation.py``` : to realize a triangulation thanks to the DLT method.
- ```lstm.py``` : to augment the model by adding some markers and improve the precision of the intial JCPs.

- ```check_build_model.py``` : to check if the biomechanical model is good defined.
- ```launch_run_ik.py``` : to have the results of the IK. <ins>You need to open Gepetto before.</ins>

  <br />
  
**Option 2 : entire pipe (from fusion_models to run_ik)**

Before to run the pipe thanks to ```main.py```, you need to open gepetto-viewer by writing ```gepetto-gui``` in the terminal.

<br />

## A little help for dependencies and installation

To be able to launch everything, you will need to follow these installations (**with conda on Linux**):

<br />

==> <ins>Create a first new environment for Mmpose only :</ins>
- MMpose is installed from source in another environment. You can follow this link for the installation : https://mmpose.readthedocs.io/en/latest/installation.html
<br />

==> <ins>Create a second environment for the pipe :</ins> 
- ```conda create --name myenv python=3.8.19 ```
- ```conda install -c conda-forge cudatoolkit=10.1.243 cudnn=7.6.5```
- ```python -m pip install -r requirements.txt```
- ```conda install meshcat-python```
- ```conda install tqdm```
- ```conda install cyipopt```
- ```conda install quadprog```

- **Pinocchio** : ```conda install -c olivier.roussel hpp-fcl example-robot-data pinocchio=2.99.0```
- **Gepetto Viewer** : ```conda install gepetto-viewer gepetto-viewer-corba -c conda-forge```

