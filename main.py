import os
from utils_triangulation import *
from utils_animation_triangul import *
from utils import *

no_sujet = int(input("Entrez le numéro du sujet (ex: 2): "))
task = input("Entrez la tâche (ex: 'assis-debout'): ")
threshold = float(input("Quelle est la valeur du threshold: "))
while True:
    affichage_anim_triangul= input("Voulez-vous afficher l'animation des résultats du LSTM ? (oui/non) : ").strip().lower()
    if affichage_anim_triangul in ['oui', 'o', 'yes', 'y']:
        afficher_resultats_triangul = True
        break
    elif affichage_anim_triangul in ['non', 'n', 'no']:
        afficher_resultats_triangul = False
        break
    else:
        print("Réponse non valide. Veuillez répondre par 'oui' ou 'non'.")


#Etape 1 : Transformation des fichiers pour ajouter les coordonées des mains à partir des fichiers de body26 et wholebody.


#Ces chemins ont été déterminés lors du process de MMPose
liste_fichiers = [
    '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26585_sujet' + str(no_sujet) + '.txt',
    '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26587_sujet' + str(no_sujet) + '.txt',
    '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26578_sujet' + str(no_sujet) + '.txt',
    '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26579_sujet' + str(no_sujet) + '.txt',
    '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26580_sujet' + str(no_sujet) + '.txt',
    '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26582_sujet' + str(no_sujet) + '.txt',
    '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26583_sujet' + str(no_sujet) + '.txt',
    '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26584_sujet' + str(no_sujet) + '.txt',
    '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26586_sujet' + str(no_sujet) + '.txt'
    ]

liste_fichiers_main = [
    '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26585_sujet' + str(no_sujet) + '.txt',
    '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26587_sujet' + str(no_sujet) + '.txt',
    '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26578_sujet' + str(no_sujet) + '.txt',
    '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26579_sujet' + str(no_sujet) + '.txt',
    '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26580_sujet' + str(no_sujet) + '.txt',
    '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26582_sujet' + str(no_sujet) + '.txt',
    '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26583_sujet' + str(no_sujet) + '.txt',
    '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26584_sujet' + str(no_sujet) + '.txt',
    '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26586_sujet' + str(no_sujet) + '.txt'
]


if not os.path.exists  ( '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/all') :
    try:    
        with open('add_hands.py') as f:
            code = f.read()
            exec(code)
            print ('Les coordonnées correspondantes aux mains ont été rajoutées.')

    except Exception as e:
        print(f"Erreur lors de l'exécution de 'add_hands.py' : {e}")

else : 
    print('Le dossier regroupant les coordonnées du corps et des mains a été récupéré.')






# Etape 2 : Triangulation


liste_fichiers_all = [
    '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/all/result_'+task+'_26585_sujet'+str(no_sujet)+'.txt',
    '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/all/result_'+task+'_26587_sujet'+str(no_sujet)+'.txt',
    '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/all/result_'+task+'_26578_sujet'+str(no_sujet)+'.txt',
    '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/all/result_'+task+'_26579_sujet'+str(no_sujet)+'.txt',
    '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/all/result_'+task+'_26580_sujet'+str(no_sujet)+'.txt',
    '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/all/result_'+task+'_26582_sujet'+str(no_sujet)+'.txt',
    '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/all/result_'+task+'_26583_sujet'+str(no_sujet)+'.txt',
    '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/all/result_'+task+'_26584_sujet'+str(no_sujet)+'.txt',
    '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/all/result_'+task+'_26586_sujet'+str(no_sujet)+'.txt'
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

for cam in donnees :
    R_extrinseque = np.zeros(shape=(3,3))
    rotation=np.array(donnees[cam]["rotation"])
    # print(rotation)
    translation=np.array([donnees[cam]["translation"]]).reshape(3,1)
    cv.Rodrigues(rotation.reshape(3,1), R_extrinseque)
    # print(R_extrinseque)
    projection = np.concatenate([R_extrinseque, translation], axis=-1)
    # print(projection)
    donnees[cam]["projection"] = projection
    # print("avec boucle pour la cam : ", cam , donnees[cam]["projection"])

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



p3ds_frames = triangulate_points_adaptive(uvs, mtxs, dists, projections, scores, threshold)
p3ds_frames = butterworth_filter(p3ds_frames,5.0)
# p3ds_frames = triangulate_points(uvs, mtxs, dists, projections)

# joints_trc = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'head', 'neck', 'hip', 'left_big_toe', 'right_big_toe', 'left_small_toe', 'right_small_toe', 'left_heel', 'right_heel','Left_hand_root', 'left_thumb1', 'left_thumb2', 'left_forefinger1', 'left_middle_finger1', 'left_ring_finger1', 'left_pinky_finger1','right_hand_root', 'right_thumb1', 'right_thumb2', 'right_forefinger1','right_middle_finger1', 'right_ring_finger1', 'right_pinky_finger1']

# # Écrire les résultats dans un fichier TRC
# with open('/home/tbousquet/Documents/Challenge/Donnees_mmpose_avec_score/jcp_coordinates_ncameras_avec_score_with_names_'+trial+'_'+subject+'_.trc', 'w') as f:
#     f.write(','.join(joints_trc) + '\n')
#     for frame in p3ds_frames:
#         frame_flat = frame.flatten()
#         f.write(','.join(map(str, frame_flat)) + '\n')

LSTM_dir = '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/post_triangulation'
output_file ='/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/post_triangulation/jcp_coordinates_ncameras_avec_score_for_LSTM_'+task+'_sujet'+str(no_sujet)+'.trc'


# Création du répertoire si nécessaire
if not os.path.exists(LSTM_dir):
    os.makedirs(LSTM_dir)
    print(f'Le répertoire {LSTM_dir} a été créé.')

# Écriture des résultats dans le fichier TRC
try:
    with open(output_file, 'w') as f:
        for frame in p3ds_frames:
            frame_flat = frame.flatten()
            f.write(','.join(map(str, frame_flat)) + '\n')
    print(f'Les coordonnées ont été écrites dans le fichier {output_file}.')
except Exception as e:
    print(f"Erreur lors de l'écriture dans le fichier {output_file} : {e}")

# Affichage de l'animation de la triangulation
if afficher_resultats_triangul:
    affichage_triangul(output_file)
else:
    print("Les résultats du LSTM ne seront pas affichés.")
