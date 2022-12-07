from datetime import datetime
import os
import time

import pandas as pd
import cv2
import numpy as np

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
    filename_df_test = pipeline.get_filename_df(labels_test, raw_data_path, force=True)
    Y_test_class, Y_test_reg, Y_test_per, Y_test_jon = pipeline.get_Ys(labels_test)
    X_test_reg = pipeline.get_X_full(filename_df_test, labels_test, Y_test_class, force=True, t="regressor")
    X_test_reg_norm, _, _ = f.normalize_features(X_test_reg, m_reg, s_reg)

    # X_test_bin = pipeline.get_X_full(filename_df_test, labels_test, Y_test_class, force=True, t="bc")
    X_test_bin = X_test_reg
    X_test_bin_norm, _, _ = f.normalize_features(X_test_bin, m_bin, s_bin)
    

    
    for index, label in labels_test.iterrows():
        # jump to the medo start frame
        medo_start = label["medo start"]
        if medo_start < 0: continue

        df = filename_df_test[label["Filename"]]

        # final predictions
        X_reg = f.extract_regressor_features(df)
        X_reg_norm, _, _ = f.normalize_features(X_reg, m_reg, s_reg)
        # X_bc = f.extract_bc_features(df)
        # X_bc_norm, _, _ = f.normalize_features(X_bc, m_bin, s_bin)
        pred_reg_final = reg_model.predict(X_reg_norm)[0]
        pred_bin_final = bin_model.predict(X_reg_norm)[0]
        # play video
        cap = cv2.VideoCapture(label["video"])
        # get fps
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps*medo_start + 20*fps)) # the data is truncated by 20 seconds
        frame_num = 0
        freq = 100
        samples_per_frame = freq / fps

        pred_reg = -1
        pred_bin = -1
        tries = 0
        while(cap.isOpened()):
            tic = time.time()
            make_prediction = frame_num % 30 == 0 and frame_num > 4*fps
            ret, frame = cap.read()
            frame_num += 1
            make_ensemble_prediction = frame_num % (30*5) == 0 and frame_num > 30*20 # only start 20 seconds in
            if ret:
                tries = 0
                if make_prediction:
                    last_sample = int(samples_per_frame * frame_num)
                    X_reg = f.extract_regressor_features(df.iloc[:last_sample,:])
                    X_reg_norm, _, _ = f.normalize_features(X_reg, m_reg, s_reg)
                    # X_bc = f.extract_bc_features(df.iloc[:last_sample,:])
                    # X_bc_norm, _, _ = f.normalize_features(X_bc, m_bin, s_bin)

                    pred_reg = reg_model.predict(X_reg_norm)[0]
                    pred_bin = bin_model.predict(X_reg_norm)[0]

                
                text_final_reg = f"CDRS: {format(pred_reg_final, '.3f')}"
                text_reg = f"CDRS: {format(pred_reg, '.3f')}"
                pos_final_reg = (int(4*frame.shape[1]/6), int(frame.shape[0]/6))
                pos_reg = (int(frame.shape[1]/6), int(frame.shape[0]/6))

                text_bin = f"Binary pred: {pred_bin}"
                text_final_bin = f"Binary pred final: {pred_bin_final}"
                pos_bin = (int(frame.shape[1]/6), int(3*frame.shape[0]/12))
                pos_final_bin = (int(4*frame.shape[1]/6), int(3*frame.shape[0]/12))
                cv2.putText(frame, text_reg, pos_reg, 0, 2, color=(255,255,255), thickness=3)
                cv2.putText(frame, text_bin, pos_bin, 0, 2, color=(255,255,255), thickness=3)
                cv2.putText(frame, text_final_reg, pos_final_reg, 0, 2, color=(255,255,255), thickness=3)
                cv2.putText(frame, text_final_bin, pos_final_bin, 0, 2, color=(255,255,255), thickness=3)


                cv2.putText(frame, f"filename: {label['Filename']}", (int(frame.shape[1]/6), int(frame.shape[0]*5/6)),0, 2, color=(255,255,255), thickness=3)

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
        

if __name__ == '__main__':
    main()

