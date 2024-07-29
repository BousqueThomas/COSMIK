import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils.utils import read_mmpose_file, get_cams_params_challenge, read_mmpose_scores, butterworth_filter
from utils.utils_animation_triangul import *
from utils.utils_triangulation import *




# no_sujet = int(input("Entrez le numéro du sujet (ex: 2): "))
# task = input("Entrez la tâche (ex: 'assis-debout'): ")
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


# Etape 2 : Triangulation


liste_fichiers = [
    f'{data_path}sujet_02/marche/body26/result_marche_cam1_sujet2.txt',
    f'{data_path}sujet_02/marche/body26/result_marche_cam2_sujet2.txt'
]

# liste_fichiers = [
#     f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_cam1_sujet' + str(no_sujet) + '.txt',
#     f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_cam2_sujet' + str(no_sujet) + '.txt'
# ]

donnees_cameras=[]
for fichier in liste_fichiers :
    donnees_cameras.append(read_mmpose_file(fichier))

print(len(donnees_cameras), 'vidéos sont traitées dans ce code.')


# Nombre de points joints (JCP)
nombre_points = 26

uvs=[]
for donnee_camera in donnees_cameras:
    uvs_camera = np.array([[ligne[2*i], ligne[2*i + 1]] for ligne in donnee_camera for i in range(nombre_points)])
    uvs_camera = uvs_camera.reshape(-1, nombre_points, 2)
    uvs.append(uvs_camera)


# Récupération des différents paramètrers propres aux caméras.
donnees = get_cams_params_challenge()

for cam in donnees :
    R_extrinseque = np.zeros(shape=(3,3))
    rotation=np.array(donnees[cam]["rotation"])
    translation=np.array([donnees[cam]["translation"]]).reshape(3,1)
    cv.Rodrigues(rotation.reshape(3,1), R_extrinseque)
    projection = np.concatenate([R_extrinseque, translation], axis=-1)
    donnees[cam]["projection"] = projection

rotations=[]
translations=[]
dists=[]
mtxs=[]
projections=[]

# Créer un dictionnaire pour stocker les données correspondant aux numéros de fichier
donnees_correspondantes = {}

# Parcourir les fichiers
for fichier in liste_fichiers:
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


scores = read_mmpose_scores(liste_fichiers)

# Traitement d'une seule frame pour la triangulation
frame_index = int(input("Entrez l'index de la frame à traiter : "))
uvs_single_frame = [uv[frame_index] for uv in uvs]

p3d_frame = triangulate_points_adaptive_single_frame(uvs_single_frame, mtxs, dists, projections, scores, threshold)

# Filtrage Butterworth 
# p3d_frame_filtered = butterworth_filter([p3d_frame], 5.0)[0]



triangul_dir = f'{data_path}sujet_02/marche/post_triangulation/single_frame'
output_file_triangul =f'{data_path}sujet_02/marche/post_triangulation/single_frame/jcp_coordinates_ncameras_avec_score_for_LSTM_marche_sujet2.trc'

# triangul_dir = f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/post_triangulation/single_frame'
# output_file_triangul =f'{data_path}sujet_0' + str(no_sujet) + '/' + task + '/post_triangulation/single_frame/jcp_coordinates_ncameras_avec_score_for_LSTM_'+task+'_sujet'+str(no_sujet)+'.trc'


# Création du répertoire si nécessaire
if not os.path.exists(triangul_dir):
    os.makedirs(triangul_dir)
    print(f'Le répertoire {triangul_dir} a été créé.')

# Écriture des résultats dans le fichier TRC
try:
    with open(output_file_triangul, 'w') as f:
        frame_flat = p3d_frame.flatten()
        f.write(','.join(map(str, frame_flat)) + '\n')
    print(f'Les coordonnées 3D ont été écrites dans le fichier suivant : jcp_coordinates_ncameras_avec_score_for_LSTM_marche_sujet2.trc.')
    # print(f'Les coordonnées 3D ont été écrites dans le fichier suivant : jcp_coordinates_ncameras_avec_score_for_LSTM_'+task+'_sujet'+str(no_sujet)+'.trc.')

except Exception as e:
    print(f"Erreur lors de l'écriture dans le fichier {output_file_triangul} : {e}")



# Etape 3 : Affichage de l'animation concernant la triangulation
if afficher_resultats_triangul:
    affichage_triangul_single_frame(output_file_triangul)
    
else:
    print("L'animation concernant la triangulation ne sera pas affichée.")

