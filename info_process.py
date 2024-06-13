from mmpose.apis import MMPoseInferencer
import os

no_sujet = 1
no_cam = 26580
task = 'marche'

# /home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_01/marche/body26/result_marche_26579_sujet1.txt'
# video_path = '/home/tbousquet/Documents/COSMIK/twirl.avi'
# out_file = '/home/tbousquet/Documents/COSMIK/result_twirl_avi.txt'

# file = f"/home/tbousquet/Documents/COSMIK/result_twirl.avi"


video_path = '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/videos/' + str(no_cam) + '/' + str(no_cam) + '.avi'

out_file_body26 = '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_' + str(no_cam) + '_sujet' + str(no_sujet) + '.txt'
file_body26 = f"/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0{str(no_sujet)}/{task}/body26/result_{task }_{str(no_cam)}_sujet{str(no_sujet)}_video_res.avi"

out_file_wholebody = '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_' + str(no_cam) + '_sujet' + str(no_sujet) + '.txt'
file_wholebody = f"/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0{str(no_sujet)}/{task}/wholebody/result_{task }_{str(no_cam)}_sujet{str(no_sujet)}_video_res.avi"

if not os.path.exists(out_file_body26):
    file=file_body26

elif not os.path.exists(out_file_wholebody): 
    file=file_wholebody


# video_path = [ 
#     # '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/videos/26585/' + str(no_cam) + '.avi',
#     '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/videos/26587/' + str(no_cam) + '.avi',
#     '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/videos/26578/' + str(no_cam) + '.avi',
# #     '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/videos/26579/' + str(no_cam) + '.avi',
# #     '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/videos/26580/' + str(no_cam) + '.avi',
# #     '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/videos/26582/' + str(no_cam) + '.avi',
# #     '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/videos/26583/' + str(no_cam) + '.avi',
# #     '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/videos/26584/' + str(no_cam) + '.avi',
# #     '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/videos/26586/' + str(no_cam) + '.avi'
# ]

# liste_fichiers = [
#     # '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26585_sujet' + str(no_sujet) + '.txt',
#     '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26587_sujet' + str(no_sujet) + '.txt',
#     '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26578_sujet' + str(no_sujet) + '.txt',
#     # '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26579_sujet' + str(no_sujet) + '.txt',
#     # '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26580_sujet' + str(no_sujet) + '.txt',
#     # '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26582_sujet' + str(no_sujet) + '.txt',
#     # '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26583_sujet' + str(no_sujet) + '.txt',
#     # '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26584_sujet' + str(no_sujet) + '.txt',
#     # '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/body26/result_' + task + '_26586_sujet' + str(no_sujet) + '.txt'
#     ]

# liste_fichiers_main = [
#     # '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26585_sujet' + str(no_sujet) + '.txt',
#     '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26587_sujet' + str(no_sujet) + '.txt',
#     '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26578_sujet' + str(no_sujet) + '.txt',
# #     '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26579_sujet' + str(no_sujet) + '.txt',
# #     '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26580_sujet' + str(no_sujet) + '.txt',
# #     '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26582_sujet' + str(no_sujet) + '.txt',
# #     '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26583_sujet' + str(no_sujet) + '.txt',
# #     '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26584_sujet' + str(no_sujet) + '.txt',
# #     '/home/tbousquet/Documents/COSMIK/Donnees challenge markerless/Data/sujet_0' + str(no_sujet) + '/' + task + '/wholebody/result_' + task + '_26586_sujet' + str(no_sujet) + '.txt'
# ]





