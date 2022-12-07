

from datetime import datetime
import os
import time

import pandas as pd
import cv2
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

import pct.preprocessing.sample_to_feature as s2f
import pct.pipeline.feature_ext as f
import pct.report_figures.regression.final_model as reg_fm
import pct.report_figures.binary_classification.final_model as bc_fm
import pct.preprocessing.parsing as parsing
import pct.pipeline.pipeline as pipeline



def main():
    # final models
    reg_model, m_reg, s_reg = reg_fm.get_final_model(test=True)
    bin_model, m_bin, s_bin = bc_fm.get_final_model(test=True)

    # for parsing test data
    labels_test = parsing.parse_test_labels()

    raw_data_path = "/Users/gustaf/Documents/skola/exjobb/tremor/data/raw_all"
    filename_df_test = pipeline.get_filename_df(labels_test, raw_data_path, force=False, t="test")
    Y_test_class, Y_test_reg, Y_test_per, Y_test_jon = pipeline.get_Ys(labels_test)
    X_test_reg = pipeline.get_X_full(filename_df_test, labels_test, Y_test_class, force=False, t="regressor")
    X_test_reg_norm, _, _ = f.normalize_features(X_test_reg, m_reg, s_reg)

    X_test_bin = pipeline.get_X_full(filename_df_test, labels_test, Y_test_class, force=False, t="bc")
    X_test_bin_norm, _, _ = f.normalize_features(X_test_bin, m_bin, s_bin)
    
    for index, label in labels_test.iterrows():
        # jump to the medo start frame
        medo_start = label["medo start"]
        if medo_start < 0: continue

        df = filename_df_test[label["Filename"]].reset_index(drop=True)

        # final predictions
        X_reg = f.extract_regressor_features(df)
        X_reg_norm, _, _ = f.normalize_features(X_reg, m_reg, s_reg)
        X_bc = f.extract_bc_features(df)
        X_bc_norm, _, _ = f.normalize_features(X_bc, m_bin, s_bin)
        pred_reg_final = reg_model.predict(X_reg_norm)[0]
        pred_bin_final = bin_model.predict(X_bc_norm)[0]
       

        pred_reg = -1
        pred_bin = -1
        tries = 0

        theta_vars = np.asarray([np.var(df.loc[i:i+100, "theta"]) for i in range(len(df) - 101)])
        
        longest_consecutive1 = f.longest_consecutive_above_threshold(theta_vars, 0.05)
        longest_consecutive2 = f.longest_consecutive_above_threshold(theta_vars, 0.5)
        print(f"longest consecutive1: {max(longest_consecutive1)}")
        print(f"longest consecutive2: {max(longest_consecutive2)}")
        exit()
        # play video
        cap = cv2.VideoCapture(label["video"])
        # get fps
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps*medo_start + 20*fps)) # the data is truncated by 20 seconds
        frame_num = 0
        freq = 100
        samples_per_frame = freq / fps
        while(cap.isOpened()):
            tic = time.time()
            make_prediction = frame_num % 30 == 0 and frame_num > 4*fps
            ret, frame = cap.read()
            frame_num += 1
            sample = int(frame_num / fps * 100)
            passed_seconds = int(np.floor(frame_num / fps))
          
            if ret:
                tries = 0
                # final predictions
                text_final_bin = f"Binary pred final: {pred_bin_final}"
                text_final_reg = f"CDRS: {format(pred_reg_final, '.3f')}"

                # theta variance
                theta_variance = format(theta_vars[sample], ".3f")

                # presenting positions
                top_right1 = (int(4*frame.shape[1]/6), int(frame.shape[0]/6))
                top_right2 = (int(4*frame.shape[1]/6), int(3*frame.shape[0]/12))
                top_left1 = (int(frame.shape[1]/6), int(frame.shape[0]/6))
                top_left2 = (int(frame.shape[1]/6), int(3*frame.shape[0]/12))
                

                
                
                # theta variance
                cv2.putText(frame, theta_variance, top_right1, 0, 2, color=(255,255,255), thickness=3)

                #final predictions
                cv2.putText(frame, text_final_reg, top_left1, 0, 2, color=(255,255,255), thickness=3)
                cv2.putText(frame, text_final_bin, top_left2, 0, 2, color=(255,255,255), thickness=3)

                # filename
                cv2.putText(frame, f"filename: {label['Filename']}", (int(frame.shape[1]/6), int(frame.shape[0]*5/6)),0, 2, color=(255,255,255), thickness=3)
                # label
                cv2.putText(frame, str(f"per: {label['per']}, jonathan: {label['jonathan']}"), (int(frame.shape[1]/6), int(frame.shape[0]*11/12)),0, 2, color=(255,255,255), thickness=3)
                toc = time.time()
                cv2.imshow('Frame', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                tries += 1
                if tries > 100:
                    break
                
            
        cap.release()
        cv2.destroyAllWindows()
        theta_vars[theta_vars > 1] = 1
        fig, ax = plt.subplots(1,2)
        ax[0].plot(theta_vars)
        ax[1].hist(theta_vars)
        plt.show()
        
        

if __name__ == '__main__':
    main()

