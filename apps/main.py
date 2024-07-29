import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils.utils_triangulation import *
from utils.utils_animation_triangul import *
from utils.utils_animation_LSTM import *
from utils.utils import *
from utils.utils_LSTM import modif_LSTM ,augmentTRC
from utils.ik_utils import *
from utils.ik_utils import *

no_sujet = int(input("Entrez le numéro du sujet (ex: 2): "))
task = input("Entrez la tâche (ex: 'assis-debout'): ")
threshold = float(input("Quelle est la valeur du threshold: "))
data_path='/home/tbousquet/Documents/Donnees_cosmik/Data/'

while True:
    affichage_anim_triangul= input("Voulez-vous afficher l'animation concernant la triangulation? (oui/non) : ").strip().lower()
    if affichage_anim_triangul in ['oui', 'o', 'yes', 'y']:
        afficher_resultats_triangul = True
        break
    elif affichage_anim_triangul in ['non', 'n', 'no']:
        afficher_resultats_triangul = False
        break
    else:
        print("Réponse non valide. Veuillez répondre par 'oui' ou 'non'.")

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





#Etape 1 : Transformation des fichiers pour ajouter les coordonées des mains à partir des fichiers de body26 et wholebody.


#Ces chemins ont été déterminés lors du process de MMPose
# liste_fichiers = [
#     f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_55555_sujet' + str(no_sujet) + '.txt',
#     f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_88888_sujet' + str(no_sujet) + '.txt'
#     ]

# liste_fichiers_main = [
#     f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_55555_sujet' + str(no_sujet) + '.txt',
#     f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_88888_sujet' + str(no_sujet) + '.txt'
# ]

liste_fichiers = [
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26585_sujet' + str(no_sujet) + '.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26587_sujet' + str(no_sujet) + '.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26578_sujet' + str(no_sujet) + '.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26579_sujet' + str(no_sujet) + '.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26580_sujet' + str(no_sujet) + '.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26582_sujet' + str(no_sujet) + '.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26583_sujet' + str(no_sujet) + '.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26584_sujet' + str(no_sujet) + '.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26586_sujet' + str(no_sujet) + '.txt'
    ]

liste_fichiers_main = [
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26585_sujet' + str(no_sujet) + '.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26587_sujet' + str(no_sujet) + '.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26578_sujet' + str(no_sujet) + '.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26579_sujet' + str(no_sujet) + '.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26580_sujet' + str(no_sujet) + '.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26582_sujet' + str(no_sujet) + '.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26583_sujet' + str(no_sujet) + '.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26584_sujet' + str(no_sujet) + '.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26586_sujet' + str(no_sujet) + '.txt'
]

if not os.path.exists  ( f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/all') :
    try:    
        with open('utils/add_hands.py') as f:
            code = f.read()
            exec(code)
            print ('Les coordonnées correspondantes aux mains ont été rajoutées.')

    except Exception as e:
        print(f"Erreur lors de l'exécution de 'add_hands.py' : {e}")

else : 
    print('Le dossier regroupant les coordonnées du corps et des mains a été récupéré.')






# Etape 2 : Triangulation


# liste_fichiers_all = [
#     f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/all/result_'+task+'_55555_sujet'+str(no_sujet)+'.txt',
#     f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/all/result_'+task+'_88888_sujet'+str(no_sujet)+'.txt'
# ]

liste_fichiers_all = [
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/all/result_'+task+'_26585_sujet'+str(no_sujet)+'.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/all/result_'+task+'_26587_sujet'+str(no_sujet)+'.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/all/result_'+task+'_26578_sujet'+str(no_sujet)+'.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/all/result_'+task+'_26579_sujet'+str(no_sujet)+'.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/all/result_'+task+'_26580_sujet'+str(no_sujet)+'.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/all/result_'+task+'_26582_sujet'+str(no_sujet)+'.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/all/result_'+task+'_26583_sujet'+str(no_sujet)+'.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/all/result_'+task+'_26584_sujet'+str(no_sujet)+'.txt',
    f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/all/result_'+task+'_26586_sujet'+str(no_sujet)+'.txt'
]


donnees_cameras=[]
for fichier in liste_fichiers_all :
    donnees_cameras.append(read_mmpose_file(fichier))

print(len(donnees_cameras), 'vidéos sont traitées dans ce code.')


# Nombre de points joints (JCP)
nombre_points = 40

uvs=[]
for donnee_camera in donnees_cameras:
    uvs_camera = np.array([[ligne[2*i], ligne[2*i + 1]] for ligne in donnee_camera for i in range(nombre_points)])
    uvs_camera = uvs_camera.reshape(-1, nombre_points, 2)
    uvs.append(uvs_camera)


# Récupération des différents paramètrers propres aux caméras.
donnees = get_cams_params_challenge()

# Calcul des matrices de projections si on a les matrices de rodrigues
for cam in donnees :
    R_extrinseque = np.zeros(shape=(3,3))
    rotation=np.array(donnees[cam]["rotation"])
    translation=np.array([donnees[cam]["translation"]]).reshape(3,1)
    cv.Rodrigues(rotation.reshape(3,1), R_extrinseque)
    projection = np.concatenate([R_extrinseque, translation], axis=-1)
    donnees[cam]["projection"] = projection


# calcul des matrices de projections si on à la matrice de rotation classique. Attention à bien laisser que les caméras sans rodrigues dans le doctionnaire (cf utils.py)
# for cam in donnees :
#     R_extrinseque = np.zeros(shape=(3,3))
#     rotation=np.array(donnees[cam]["rotation"]).reshape(3,3)
#     translation=np.array([donnees[cam]["translation"]]).reshape(3,1)
#     projection = np.concatenate([rotation, translation], axis=-1)
#     donnees[cam]["projection"] = projection




rotations=[]
translations=[]
dists=[]
mtxs=[]
projections=[]

# Créer un dictionnaire pour stocker les données correspondant aux numéros de fichier
donnees_correspondantes = {}

# Parcourir les fichiers
for fichier in liste_fichiers_all:
    # Extraire le numéro de série de la caméra
    numero_fichier = fichier.split('_')[-2]
    
    # Vérifier si le numéro de fichier correspond à une entrée dans le dictionnaire de données
    if numero_fichier in donnees:
        donnees_correspondantes[fichier] = donnees[numero_fichier]
        # Ajouter la matrice mtx à la liste mtxs
        mtxs.append(donnees[numero_fichier]["mtx"])
        dists.append(donnees[numero_fichier]["dist"])
        translations.append(donnees[numero_fichier]["translation"])
        rotations.append(donnees[numero_fichier]["rotation"])
        projections.append(donnees[numero_fichier]["projection"])


scores = read_mmpose_scores(liste_fichiers_all)



p3ds_frames = triangulate_points_adaptive(uvs, mtxs, dists, projections, scores, threshold)
p3ds_frames = butterworth_filter(p3ds_frames,5.0)

# joints_trc = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'head', 'neck', 'hip', 'left_big_toe', 'right_big_toe', 'left_small_toe', 'right_small_toe', 'left_heel', 'right_heel','Left_hand_root', 'left_thumb1', 'left_thumb2', 'left_forefinger1', 'left_middle_finger1', 'left_ring_finger1', 'left_pinky_finger1','right_hand_root', 'right_thumb1', 'right_thumb2', 'right_forefinger1','right_middle_finger1', 'right_ring_finger1', 'right_pinky_finger1']

# # Écrire les résultats dans un fichier TRC
# with open('/home/tbousquet/Documents/Challenge/Donnees_mmpose_avec_score/jcp_coordinates_ncameras_avec_score_with_names_'+trial+'_'+subject+'_.trc', 'w') as f:
#     f.write(','.join(joints_trc) + '\n')
#     for frame in p3ds_frames:
#         frame_flat = frame.flatten()
#         f.write(','.join(map(str, frame_flat)) + '\n')

triangul_dir = f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/post_triangulation'
output_file_triangul =f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/post_triangulation/jcp_coordinates_ncameras_avec_score_for_LSTM_'+task+'_sujet'+str(no_sujet)+'.trc'


# Création du répertoire si nécessaire
if not os.path.exists(triangul_dir):
    os.makedirs(triangul_dir)
    print(f'Le répertoire {triangul_dir} a été créé.')

# Écriture des résultats dans le fichier TRC
try:
    with open(output_file_triangul, 'w') as f:
        for frame in p3ds_frames:
            frame_flat = frame.flatten()
            f.write(','.join(map(str, frame_flat)) + '\n')
    print(f'Les coordonnées 3D ont été écrites dans le fichier suivant : jcp_coordinates_ncameras_avec_score_for_LSTM_'+task+'_sujet'+str(no_sujet)+'.trc.')
except Exception as e:
    print(f"Erreur lors de l'écriture dans le fichier {output_file_triangul} : {e}")




# Etape 3 : Affichage de l'animation concernant la triangulation
if afficher_resultats_triangul:
    affichage_triangul(output_file_triangul)
    
else:
    print("L'animation concernant la triangulation ne sera pas affichée.")



# Etape 4: LSTM

modif_LSTM(output_file_triangul,no_sujet,task,data_path)
pathInputTRCFile=f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/LSTM/jcp_coordinates_ncameras_transformed_'+task+'_'+str(no_sujet)+'.trc'
    # pathInputTRCFile="/home/tbousquet/Documents/LSTM/data/7272a71a-e70a-4794-a253-39e11cb7542c/PreAugmentation/a9fd6740-1c9d-40df-beca-15e6eecf08d7.trc"
if no_sujet == 1:
    subject_mass=60.0
    subject_height=1.57
elif no_sujet == 2:
    subject_mass=58.0
    subject_height=1.74
elif no_sujet == 3:
    subject_mass=66.0
    subject_height=1.83

pathOutputTRCFile=f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/LSTM/jcp_coordinates_ncameras_augmented_'+task+'_'+str(no_sujet)+'.trc'
pathOutputCSVFile = f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/LSTM/jcp_coordinates_ncameras_augmented_'+task+'_'+str(no_sujet)+'.csv'
augmenterDir=os.getcwd()
augmentTRC(pathInputTRCFile, subject_mass, subject_height, pathOutputTRCFile, pathOutputCSVFile, augmenterDir, augmenterModelName="LSTM", augmenter_model='v0.3', offset=True)


# Etape 5: Animation du LSTM :
if afficher_resultats_LSTM:
    affichage_LSTM(pathOutputCSVFile)
else:
    print("L'animation concernant le LSTM ne sera pas affichée.")


# Etape 6 : Inverse Kinematics
print("L'IK va commencer.")

# pour l'ik aveec ipopt :
# ik_pipe (no_sujet, task, data_path)

# pour l'ik qp :
ik_pipe_qp (no_sujet, task, data_path)
