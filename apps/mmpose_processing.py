import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer
import os
from tqdm import tqdm


#Lancer mmpose_processing pour toutes les vidéos que l'on souhaite utiliser !!! 


def process_video_save(video_path, inferencer, output_txt_path):
    """Process a video frame by frame, visualize predicted keypoints, and save keypoints to a .txt file."""

    # Initialize video capture

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4)) 
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    size = (frame_width, frame_height)
    # print("size")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(file, fourcc, 30.0, size)


    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Open the output .txt file for writing keypoints
    with open(output_txt_path, 'w') as f:
        i=0
        with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:

            while True:
                ret, frame = cap.read()
                # print("ret", ret)
                if not ret:
                    
                    break
                
                # Convert frame to RGB as inferencer expects RGB input
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run inference
                result_generator = inferencer(frame_rgb, return_vis=True,draw_bbox = True, show=False)
                result = next(result_generator)
        
                # Extract the visualization result
                vis_frame = result['visualization'][0]
                
                # Convert visualization result from RGB back to BGR for OpenCV
                vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
                out.write(vis_frame_bgr)

                # Display the frame with inferred keypoints
                cv2.imshow('Video with Keypoints', vis_frame_bgr)
                
                # Extract keypoints
                predictions = result['predictions']
                for prediction in predictions:
                    
                    # print("prediction", prediction)
                    
                    # bbox1 = np.array(prediction[0]['bbox'])
                    # bbox2 = np.array(prediction[1]['bbox'])

                    keypoints1 = np.asarray(prediction[0]['keypoints'])
                    # print(keypoints1)
                    
                    # keypoints2 = np.asarray(prediction[1]['keypoints'])

                    # print(np.shape(keypoints1))
                    if keypoints1 is not None:
                        # print(keypoints)
                        keypoints1_2d = keypoints1[:, :2]  # We need only the x, y coordinates
                        # print("keypoints", keypoints1_2d)
                        score = prediction[0]['bbox_score']
                        # print("score", prediction[0]['bbox_score'])
                        # Flatten the keypoints array and convert to string with comma separator

                        keypoints1_str = ','.join(map(str, keypoints1_2d.flatten()))
                        # keypoints1_str = str(i) + "," + keypoints1_str
                        score_str = ','.join(map(str, score.flatten()))
                        # score_str = "," + score_str
                        
                        
                        # Combine keypoints and scores into one string
                        combined_str = str(i) + "," + score_str + "," + keypoints1_str

                        # Write the keypoints to the .txt file
                        f.write(f"{combined_str}\n")

                # Update the progress bar
                pbar.update(1)

                # Press 'q' to exit the loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                i += 1



    # Release the video capture and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()


while True:
    try:
        no_sujet = int(input("Entrez le numéro du sujet (ex: 2): "))
        no_cam = int(input("Entrez le numéro de la caméra (ex: 26578): "))
        task = input("Entrez la tâche (ex: 'assis-debout'): ")

        data_path='/home/tbousquet/Documents/Donnees_cosmik/Data/'

        if not os.path.exists(f"{data_path}sujet_0"+str(no_sujet)+"/"+task+"/body26"):
            os.makedirs(f"{data_path}sujet_0"+str(no_sujet)+"/"+task+"/body26")
            print(f'Le répertoire pour body26 a été créé.')

        if not os.path.exists(f"{data_path}sujet_0"+str(no_sujet)+"/"+task+"/wholebody"):
            os.makedirs(f"{data_path}sujet_0"+str(no_sujet)+"/"+task+"/wholebody")
            print(f'Le répertoire pour wholebody a été créé.')

        video_path = f'{data_path}sujet_0{no_sujet}/{task}/videos/{no_cam}/{no_cam}.avi'
        out_file_body26 = f'{data_path}sujet_0{no_sujet}/{task}/body26/result_{task}_{no_cam}_sujet{no_sujet}.txt'
        file_body26 = f'{data_path}sujet_0{no_sujet}/{task}/body26/result_{task}_{no_cam}_sujet{no_sujet}_video_res.avi'
        out_file_wholebody = f'{data_path}sujet_0{no_sujet}/{task}/wholebody/result_{task}_{no_cam}_sujet{no_sujet}.txt'
        file_wholebody = f'{data_path}sujet_0{no_sujet}/{task}/wholebody/result_{task}_{no_cam}_sujet{no_sujet}_video_res.avi'



        if not os.path.exists(out_file_body26):
            file = file_body26
            inferencer = MMPoseInferencer('body26')
            print('Processing de MMPose avec le modèle body26.')
            process_video_save(video_path, inferencer, out_file_body26)
        else:
            print('Le fichier résultat MMPose avec le modèle body26 a été récupéré.')

        if not os.path.exists(out_file_wholebody):
            file = file_wholebody
            inferencer = MMPoseInferencer('wholebody')
            print('Processing de MMPose avec le modèle wholebody.')
            process_video_save(video_path, inferencer, out_file_wholebody)
        else:
            print('Le fichier résultat MMPose avec le modèle wholebody a été récupéré.')

        retry = input("Voulez-vous faire un autre essai (oui/non) ? ").strip().lower()
        if retry != 'oui':
            print("Fin du programme.")
            break

    except Exception as e:
        print(f"Erreur : {e}")
        print("Veuillez réessayer.")
