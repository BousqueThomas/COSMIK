from utils_LSTM import modif_LSTM ,augmentTRC
from utils_animation_LSTM import *
import os

no_sujet = int(input("Entrez le numéro du sujet (ex: 2): "))
task = input("Entrez la tâche (ex: 'assis-debout'): ")

data_path='/home/tbousquet/Documents/Donnees_cosmik/Data/'
output_file_triangul =f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/post_triangulation/jcp_coordinates_ncameras_avec_score_for_LSTM_'+task+'_sujet'+str(no_sujet)+'.trc'

while True:
    affichage_anim_LSTM= input("Voulez-vous afficher l'animation concernant le LSTM ? (oui/non) : ").strip().lower()
    if affichage_anim_LSTM in ['oui', 'o', 'yes', 'y']:
        afficher_resultats_LSTM = True
        break
    elif affichage_anim_LSTM in ['non', 'n', 'no']:
        afficher_resultats_LSTM = False
        break
    else:
        print("Réponse non valide. Veuillez répondre par 'oui' ou 'non'.")


# Etape 4: LSTM


if not os.path.exists(f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/LSTM/'):
    os.makedirs(f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/LSTM/')
    print(f'Le répertoire pour le LSTM a été créé.')

modif_LSTM(output_file_triangul,no_sujet,task)
pathInputTRCFile=f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/LSTM/jcp_coordinates_ncameras_transformed_'+task+'_'+str(no_sujet)+'.trc'
if no_sujet == 1:
    subject_mass=60.0
    subject_height=1.57
elif no_sujet == 2:
    subject_mass=58.0
    subject_height=1.74
    pathOutputTRCFile=f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/LSTM/jcp_coordinates_ncameras_augmented_'+task+'_'+str(no_sujet)+'.trc'
    pathOutputCSVFile = f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/LSTM/jcp_coordinates_ncameras_augmented_'+task+'_'+str(no_sujet)+'.csv'
    augmenterDir=os.getcwd()
augmentTRC(pathInputTRCFile, subject_mass, subject_height, pathOutputTRCFile, pathOutputCSVFile, augmenterDir, augmenterModelName="LSTM", augmenter_model='v0.3', offset=True)


# Etape 5: Animation du LSTM :
if afficher_resultats_LSTM:
    affichage_LSTM(pathOutputCSVFile)
else:
    print("L'animation concernant le LSTM ne sera pas affichée.")