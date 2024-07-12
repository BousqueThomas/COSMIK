import pinocchio as pin 
import numpy as np
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils.read_write_utils import plot_joint_angle_results

subject = 'sujet_02'
task = 'marche'
data_path='/home/tbousquet/Documents/Donnees_cosmik/Data/'

results_directory = "./results_IK/"+subject+"/"+task
# # Plots 
plot_joint_angle_results(results_directory)