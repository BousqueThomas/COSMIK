# COSMIK
Welcome on the COSMIK repository

This repo gives you a pipeline that permits you to have Inverse Kinematics thanks to basic videos from as minimum two cameras.

You can run a code for each step of the pipe : 

- mmpose_processing : to have JCPs from a video. You need to process this code for each cameras and each video.
- fusion_models : to concatenate the results of two mmpose models (body26 and wholebody)
- triangulation : to realize a triangulation thanks to the DLT method.
- lstm : to augment the model and add some markers

- check_build_model : to check if the biomechanical model is ok
- launch_run_ik : to have the results of the IK

Or you can run main.py to do the entire pipe from fusion_models to run_ik.




To be able to launch everything, you will need the following installations :

- Create a new environment for Mmpose 
- MMpose is installed from source in another environment. You can follow this link for the installation : https://mmpose.readthedocs.io/en/latest/installation.html

- Create a second environment for the pipe : conda create --name myenv python=3.8.19
- conda install -c conda-forge cudatoolkit=10.1.243 cudnn=7.6.5
- python -m pip install -r requirements.txt

- conda install meshcat-python
- conda install tqdm
- conda install cyipopt
- conda install quadprog

- Pinocchio : conda install -c olivier.roussel hpp-fcl example-robot-data pinocchio=2.99.0
- Gepetto Viewer : conda install gepetto-viewer gepetto-viewer-corba -c conda-forge

Before to run the pipe or the main.py, you need to open gepetto-viewer by writing "gepetto-gui" in the terminal.

