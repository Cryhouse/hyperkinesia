import cv2
import pandas as pd

import pct.preprocessing.parsing as parsing

import binary_classifier as bc
import regressor as reg
import feature_ext as f
import pipeline

from pct.config import RAW_DATA_DIR, LABELS_PATH, TRAIN_PATIENTS, TEST_PATIENTS

def display_video(video_path, df, svm, svr, m, s, jonathan, per, train):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    pred_bin = bc.predict_bin(df, svm, m, s)
    pred_reg = reg.predict_regressor(df, svr, m, s)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps*20))
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        width = frame.shape[1]
        height = frame.shape[0]
        pos = (int(width/6), int(height/6))
        pos1 = (int(4*width/6), int(height/6))
        pos2 = (int(width/6), int(height*5/6))
        pos3 = (int(width/6), int(height*11/12))

        text = ""
        color = (255,255,255)
        if (pred_bin == 0) == ((jonathan + per) == 0):
            text = "correct"
            color = (30,255,30)
        elif (jonathan + per)/2 > 1.5 and pred_bin == 0:
            text = "severly false negative"
            color = (255,30,30)
        else:
            text = "incorrect"
            color = (255,255,0)
        
        cv2.putText(frame, 'train' if train else 'test', (int(width/6), int(height/12)), 0, 2, color=(255,255,255), thickness=3)

        cv2.putText(frame, f'binary prediction {pred_bin}', pos, 0, 2, color=(255,255,255), thickness=3)
        cv2.putText(frame, text, pos1, 0, 2, color=color, thickness=3)
        cv2.putText(frame, f'jonathan {jonathan}', pos2, 0, 2, color=(255,255,255), thickness=3)
        cv2.putText(frame, f'per {per}', pos3, 0, 2, color=(255,255,255), thickness=3)

        cv2.putText(frame, f"CDRS prediction: {pred_reg[0]}", (int(width/6), int(3*height/12)), 0, 2, color=(255,255,255), thickness=3)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()



def main():
    labels = parsing.parse_labels_minus_test()
    filename_df = pipeline.get_filename_df(labels, RAW_DATA_DIR)

    Y_full_class, Y_full_reg, Y_full_per, Y_full_jon = pipeline.get_Ys(labels)
    X_full = pipeline.get_X_full(filename_df, labels, Y_full_class, force=False)
    svm, best_m, best_s, train_inds, val_inds = bc.get_best_model(X_full, Y_full_class, plot=True, N=1000)

    # nonzero_mask = Y_full_reg > 0
    # Y_full_nonzero = Y_full_reg[nonzero_mask]
    # X_full_nonzero = X_full[nonzero_mask,:]
    svr, best_m, best_s, best_val_corr, best_train_inds, best_val_inds = reg.get_best_SVR(X_full, Y_full_reg, plot=True)

    for i, (index, row) in enumerate(labels.iterrows()):
        if index in train_inds: continue
        filename = row["Filename"]
        if filename not in filename_df.keys(): continue
        video = row["video"]
        
        df = filename_df[filename]
        #pred = bc.predict_bin(df, svm, best_m, best_s)
        display_video(video, df, svm, svr, best_m, best_s, row["jonathan"], row["per"], index in train_inds)


if __name__ == '__main__':
    main()

