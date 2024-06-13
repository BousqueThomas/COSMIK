from info_process import *
from mmpose_processing import process_video_save
import os
from triangulation import *

#MMPose
if not os.path.exists(out_file_body26):
    print('Processing de MMPose avec le modèle body26.')
    inferencer = MMPoseInferencer('body26') 
    file=file_body26
    process_video_save(video_path, inferencer, out_file_body26)
else :
    print('Le fichier résultat MMPose avec le modèle body26 a été récupéré.')

if not os.path.exists(out_file_wholebody):
    print('Processing de MMPose avec le modèle wholebody')
    inferencer = MMPoseInferencer('wholebody') 
    file=file_wholebody
    process_video_save(video_path, inferencer, out_file_wholebody)
else :
    print('Le fichier résultat MMPose avec le modèle wholebody a été récupéré.')


#Ajout des mains sur le fichier txt

#Ajout du score sur les fichiers .txt de MMPose.

#Triangulation





# for index,fichier in enumerate(liste_fichiers):
#     if not os.path.exists(fichier):
#         print('Processing de MMPose avec le modèle body26.')
#         inferencer = MMPoseInferencer('body26') 
#         file=f"/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0{str(no_sujet)}/{task}/body26/result_{task }_{str(index)}_sujet{str(no_sujet)}_video_res.avi"

#         process_video_save(file, video_path[index], inferencer, out_file_body26)
#     else :
#         print('Le fichier résultat MMPose avec le modèle body26 a été récupéré.')

# for index,fichier in enumerate(liste_fichiers_main):
#     if not os.path.exists(fichier):
#         print('Processing de MMPose avec le modèle wholebody.')
#         inferencer = MMPoseInferencer('wholebody') 
#         file=f"/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0{str(no_sujet)}/{task}/wholebody/result_{task }_{str(index)}_sujet{str(no_sujet)}_video_res.avi"

#         process_video_save(file, video_path[index], inferencer, out_file_wholebody)
#     else :
#         print('Le fichier résultat MMPose avec le modèle wholebody a été récupéré.')
