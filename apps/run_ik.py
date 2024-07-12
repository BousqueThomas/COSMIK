import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils.ik_utils import ik_pipe


no_sujet = int(input("Entrez le numéro du sujet (ex: 2): "))
task = input("Entrez la tâche (ex: 'assis-debout'): ")
data_path='/home/tbousquet/Documents/Donnees_cosmik/Data/'
ik_pipe (no_sujet, task, data_path)