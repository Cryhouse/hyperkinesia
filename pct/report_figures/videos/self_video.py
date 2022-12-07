from datetime import datetime
import os
import time

import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

import pct.preprocessing.sample_to_feature as s2f
import pct.pipeline.feature_ext as f
import pct.report_figures.regression.final_model as reg_fm
import pct.report_figures.binary_classification.final_model as bc_fm


def decorate_frame(frame, theta_var, pred_reg, pred_bin):
    # positions
    # top
    top_right1 = (int(4*frame.shape[1]/6), int(frame.shape[0]/6))
    top_right2 = (int(4*frame.shape[1]/6), int(3*frame.shape[0]/12))
    top_left1 = (int(frame.shape[1]/6), int(frame.shape[0]/6))
    top_left2 = (int(frame.shape[1]/6), int(3*frame.shape[0]/12))
    

    # bottom
    bot_left1 = (int(frame.shape[1]/6), int(frame.shape[0]*5/6))
    bot_left2 = (int(frame.shape[1]/6), int(frame.shape[0]*11/12))
    bot_right1 = (4*int(frame.shape[1]/6), int(frame.shape[0]*5/6))

    # regression predictions (top left)
    text_reg = f"CDRS: {format(pred_reg, '.3f')}"
    cv2.putText(frame, text_reg, top_left1, 0, 2, color=(255,255,255), thickness=3)


    # binary predictions (top right)
    text_bin = f"Bin: {format(pred_bin, '.3f')}"
    cv2.putText(frame, text_bin, top_right1, 0, 2, color=(255,255,255), thickness=3)
    theta_var_text = f"theta var: {format(theta_var, '.3f')}"
    cv2.putText(frame, theta_var_text, bot_right1, 0, 2, color=(255,255,255), thickness=3)

    return frame

def main():
    movies = [ 
        "/Users/gustaf/Documents/skola/exjobb/tremor/video/normal.MOV",
        "/Users/gustaf/Documents/skola/exjobb/tremor/video/hyper.MOV",
        "/Users/gustaf/Documents/skola/exjobb/tremor/video/activities.MOV",
    ]
    data_files = [
        "/Users/gustaf/Downloads/TREMOR12_samples_2022_11_29_1134.csv",
        "/Users/gustaf/Documents/skola/exjobb/tremor/video/normal.csv",
        "/Users/gustaf/Documents/skola/exjobb/tremor/video/hyper.csv",
        # "/Users/gustaf/Documents/skola/exjobb/tremor/video/activities.csv",
    ]
    reg_model, m_reg, s_reg, train_inds, val_inds, filename_df, labels = reg_fm.get_final_model2()
    bin_model = bc_fm.get_final_model2(train_inds, val_inds)


    for movie, data_file in zip(movies, data_files):
        # get the model
        
        # correct timestamp of datafile
        df = pd.read_csv(data_file)

        df = df[["accX", "accY", "accZ", "gravX", "gravY", "gravZ"]]
        df.rename(columns= {"accX": "linaccX", "accY": "linaccY", "accZ": "linaccZ"}, inplace=True)
        angles = s2f.grav2angles(df[["gravX", "gravY", "gravZ"]].to_numpy().T)
        df["theta"] = angles[0,:]
        df["phi"] = angles[1,:]
        
        timestamp = int(df.iloc[0,0]/1000)
        
        ref_time = datetime(2001, 1, 1, 0, 0, 0, 0).timestamp()
        timestamp_corrected = ref_time + timestamp
        
        dt = datetime.fromtimestamp(timestamp_corrected)

        # get timestamp of video
        video_creation = os.stat(movie).st_birthtime
        video_creation = os.path.getmtime(movie)

        video_creation_datetime = datetime.fromtimestamp(video_creation)


        print(f"video creation datetime: {video_creation_datetime}")
        print(f"first timestamp data file: {dt}")
        
        # play video
        cap = cv2.VideoCapture(movie)
        # get fps
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_num = 0
        freq = 100
        samples_per_frame = freq / fps

        pred_reg = -1
        pred_bin = -1
        theta_var = -1
        theta_vars = []
        while(cap.isOpened()):
            tic = time.time()
            make_prediction = frame_num % 30 == 0 and frame_num > 4*fps
            ret, frame = cap.read()
            frame_num += 1
            last_sample = frame_num * 100 / fps

            if ret:
                if make_prediction:
                    last_sample = int(samples_per_frame * frame_num)
                    X_reg = f.extract_regressor_features(df.iloc[:last_sample,:])
                    X_reg_norm, _, _ = f.normalize_features(X_reg, m_reg, s_reg)

                    pred_reg = reg_model.predict(X_reg_norm)[0]
                    pred_bin = bin_model.predict(X_reg_norm)[0]

                if last_sample > 200:
                    theta_var = np.var(df.loc[last_sample-200:last_sample, "theta"])
                    theta_vars.append(theta_var)
                
                frame = decorate_frame(frame, theta_var, pred_reg, pred_bin)
                cv2.imshow('Frame', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
                
            
        cap.release()
        cv2.destroyAllWindows()
        plt.plot(theta_vars)
        plt.title("Theta variances")
        plt.grid(True, color="#93a1a1", alpha=0.3)
        plt.show()
        

if __name__ == '__main__':
    main()

