import cv2
import numpy as np
from info_process import *




def process_video_save(video_path, inferencer, output_txt_path):
    """Process a video frame by frame, visualize predicted keypoints, and save keypoints to a .txt file."""

    # Initialize video capture

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4)) 
    
    size = (frame_width, frame_height)
    print("size")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(file, fourcc, 30.0, size)


    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Open the output .txt file for writing keypoints
    with open(output_txt_path, 'w') as f:
        i=0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to RGB as inferencer expects RGB input
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference
            result_generator = inferencer(frame_rgb, return_vis=True, show=False)
            result = next(result_generator)

            # Extract the visualization result
            vis_frame = result['visualization'][0]

            # Convert visualization result from RGB back to BGR for OpenCV
            vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)

            out.write(vis_frame_bgr)

            # Display the frame with inferred keypoints
           # cv2.imshow('Video with Keypoints', vis_frame_bgr)
            
            # Extract keypoints
            predictions = result['predictions']
            for prediction in predictions:
                # print(prediction[0]['keypoints'])
                keypoints = np.asarray(prediction[0]['keypoints'])
                print(np.shape(keypoints))
                if keypoints is not None:
                    keypoints_2d = keypoints[:, :2]  # We need only the x, y coordinates

                    # Flatten the keypoints array and convert to string with comma separator
                    keypoints_str = ','.join(map(str, keypoints_2d.flatten()))
                    keypoints_str = str(i) + "," + keypoints_str
                    
                    # Write the keypoints to the .txt file
                    f.write(f"{keypoints_str}\n")
            
            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            i += 1

    # Release the video capture and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()
