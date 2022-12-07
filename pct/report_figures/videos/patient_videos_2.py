
import os
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

import pct.pipeline.feature_ext as f
import pct.report_figures.regression.final_model as reg_fm
import pct.report_figures.binary_classification.final_model as bc_fm


def decorate_frame(frame, label, theta_var, pred_reg_final, pred_reg, pred_bin_final, pred_bin):
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
    text_final_reg = f"CDRS final: {format(pred_reg_final, '.3f')}"
    text_reg = f"CDRS: {format(pred_reg, '.3f')}"
    cv2.putText(frame, text_reg, top_left1, 0, 2, color=(255,255,255), thickness=3)
    cv2.putText(frame, text_final_reg, top_left2, 0, 2, color=(255,255,255), thickness=3)

    # binary predictions (top right)
    text_final_bin = f"Bin final: {format(pred_bin_final, '.3f')}"
    text_bin = f"Bin: {format(pred_bin, '.3f')}"
    cv2.putText(frame, text_bin, top_right1, 0, 2, color=(255,255,255), thickness=3)
    cv2.putText(frame, text_final_bin, top_right2, 0, 2, color=(255,255,255), thickness=3)
    theta_var_text = f"theta var: {format(theta_var, '.3f')}"
    cv2.putText(frame, theta_var_text, bot_right1, 0, 2, color=(255,255,255), thickness=3)

    cv2.putText(frame, f"filename: {label['Filename']}", bot_left1,0, 2, color=(255,255,255), thickness=3)
    cv2.putText(frame, str(f"per: {label['per']}, jonathan: {label['jonathan']}"), bot_left2 ,0, 2, color=(255,255,255), thickness=3)
    return frame

def save_video(df, label, reg_model, bin_model, m, s):
    medo_start = label["medo start"]
    if medo_start < 0: return
    X_reg = f.extract_regressor_features(df)
    X_reg_norm, _, _ = f.normalize_features(X_reg, m, s)
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
    theta_var = -1

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    size = (frame_width, frame_height)
    
    # Below VideoWriter object will create
    # a frame of above defined The output 
    # is stored in 'filename.avi' file.
    filename = ""
    plot_filename = ""
    for i in range(100):
        proposal = str(i).zfill(3) + ".avi"
        if proposal not in os.listdir(os.path.join(os.getcwd(), "patients")):
            filename = proposal
            plot_filename = proposal[:-3] + "png"
            break
    result = cv2.VideoWriter(os.path.join("patients", filename), 
                            cv2.VideoWriter_fourcc(*'DIVX'),
                            fps, size)

    tries = 0
    theta_vars = []
    while(cap.isOpened()):
        tic = time.time()
        make_prediction = frame_num % 30 == 0 and frame_num > 4*fps
        ret, frame = cap.read()
        frame_num += 1
        last_sample = int(samples_per_frame * frame_num)
        make_ensemble_prediction = frame_num % (30*5) == 0 and frame_num > 30*20 # only start 20 seconds in
        if ret:
            tries = 0
            if make_prediction:
                X_reg = f.extract_regressor_features(df.iloc[:last_sample,:])
                X_reg_norm, _, _ = f.normalize_features(X_reg, m, s)
                # X_bc = f.extract_bc_features(df.iloc[:last_sample,:])
                # X_bc_norm, _, _ = f.normalize_features(X_bc, m_bin, s_bin)
                pred_reg = reg_model.predict(X_reg_norm)[0]
                pred_bin = bin_model.predict(X_reg_norm)[0]
            if last_sample > 200:
                theta_var = np.var(df.loc[last_sample-200:last_sample, "theta"])
                theta_vars.append(theta_var)
            frame = decorate_frame(frame, label, theta_var, pred_reg_final, pred_reg, pred_bin_final, pred_bin)
            result.write(frame)

        else:
            tries += 1
            if tries > 100:
                break
            
        
    cap.release()
    cv2.destroyAllWindows()
    fig, ax = plt.subplots()
    ax.plot(theta_vars)
    fig.suptitle("Theta variances")
    ax.grid(True, color="#93a1a1", alpha=0.3)
    fig.savefig(plot_filename)

def main():
    # final models
    reg_model, m_reg, s_reg, val_inds, train_inds, filename_df, labels = reg_fm.get_final_model2()
    bin_model = bc_fm.get_final_model2(train_inds, val_inds)
    val_labels = labels.iloc[val_inds, :]
    # Y_val_class, Y_val_reg, Y_val_per, Y_val_jon = pipeline.get_Ys(val_labels)
    # X_val = pipeline.get_X_full(filename_df, val_labels, Y_val_reg)
    # X_val_norm, m, s = f.normalize_features(X_val, m_reg, s_reg)
    

    
    for index, label in val_labels.iterrows():
        
        # jump to the medo start frame
        medo_start = label["medo start"]
        if medo_start < 0: continue

        df = filename_df[label["Filename"]]

        tic = time.time()
        save_video(df, label, reg_model, bin_model, m_reg, s_reg)
        toc = time.time()
        print(f"time for saving video: {toc - tic}")
        continue
        
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
        theta_var = -1

        tries = 0
        theta_vars= []
        while(cap.isOpened()):
            tic = time.time()
            make_prediction = frame_num % 30 == 0 and frame_num > 4*fps
            ret, frame = cap.read()
            frame_num += 1
            last_sample = int(samples_per_frame * frame_num)
            make_ensemble_prediction = frame_num % (30*5) == 0 and frame_num > 30*20 # only start 20 seconds in
            if ret:
                tries = 0
                if make_prediction:
                    X_reg = f.extract_regressor_features(df.iloc[:last_sample,:])
                    X_reg_norm, _, _ = f.normalize_features(X_reg, m_reg, s_reg)
                    # X_bc = f.extract_bc_features(df.iloc[:last_sample,:])
                    # X_bc_norm, _, _ = f.normalize_features(X_bc, m_bin, s_bin)
                    pred_reg = reg_model.predict(X_reg_norm)[0]
                    pred_bin = bin_model.predict(X_reg_norm)[0]
                if last_sample > 200:
                    theta_var = np.var(df.loc[last_sample-200:last_sample, "theta"])
                    theta_vars.append(theta_var)
                # text_final_reg = f"CDRS: {format(pred_reg_final, '.3f')}"
                # text_reg = f"CDRS: {format(pred_reg, '.3f')}"
                # pos_final_reg = (int(4*frame.shape[1]/6), int(frame.shape[0]/6))
                # pos_reg = (int(frame.shape[1]/6), int(frame.shape[0]/6))

                # pos_bin = (int(frame.shape[1]/6), int(3*frame.shape[0]/12))
                # pos_final_bin = (int(4*frame.shape[1]/6), int(3*frame.shape[0]/12))
                # cv2.putText(frame, text_reg, pos_reg, 0, 2, color=(255,255,255), thickness=3)
                # cv2.putText(frame, text_final_reg, pos_final_reg, 0, 2, color=(255,255,255), thickness=3)

                # theta_var_text = f"theta var: {format(theta_var, '.3f')}"
                # cv2.putText(frame, theta_var_text, pos_bin, 0, 2, color=(255,255,255), thickness=3)

                # cv2.putText(frame, f"filename: {label['Filename']}", (int(frame.shape[1]/6), int(frame.shape[0]*5/6)),0, 2, color=(255,255,255), thickness=3)

                # cv2.putText(frame, str(f"per: {label['per']}, jonathan: {label['jonathan']}"), (int(frame.shape[1]/6), int(frame.shape[0]*11/12)),0, 2, color=(255,255,255), thickness=3)
                # toc = time.time()
                frame = decorate_frame(frame, label, theta_var, pred_reg_final, pred_reg, pred_bin_final, pred_bin)
                cv2.imshow('Frame', frame)
                k = cv2.waitKey(25)
                if k & 0xFF == ord('q'):
                    break
                elif k & 0xFF == ord('s'):
                    tic = time.time()
                    save_video(df, label, reg_model, bin_model, m_reg, s_reg)
                    toc = time.time()
                    print(f"time for saving video: {toc - tic}")
                    break

            else:
                tries += 1
                if tries > 100:
                    break
                
            
        cap.release()
        cv2.destroyAllWindows()
        plt.plot(theta_vars)
        plt.title("Theta variances")
        plt.grid(True, color="#93a1a1", alpha=0.3)
        plt.show()

if __name__ == '__main__':
    main()

