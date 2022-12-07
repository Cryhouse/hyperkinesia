import numpy as np
from prettytable import PrettyTable

import pct.find_orientation.animate_orientation as ao
import pct.preprocessing.data_parsing as data_parsing
import pct.preprocessing.get_orientation as go


def make_video():
    apple_data_path = "/Users/gustaf/Documents/skola/exjobb/tremor/data/orientation_test/apple_test.csv"
    apple_data = data_parsing.read_apple(apple_data_path)
    apple_grav = apple_data.loc[:,["gravX", "gravY", "gravZ"]].to_numpy().T

    # add linear acceleration to the apple data
    apple_data_noised = apple_data.loc[:,["gravX", "gravY", "gravZ"]].to_numpy().T * 9.82 + apple_data.loc[:,["accX", "accY", "accZ"]].to_numpy().T
    # extract orientation from apple_data_noised
    grav_est = go.get_orientation_naive(apple_data_noised)
    grav_est_norm = np.array([item / np.sum(item**2)**0.5 for item in grav_est.T]).T

    loss = np.sum(np.sum(apple_grav - grav_est_norm, axis=0)**2)

    ao.animate_orientation(apple_grav, "Apple gravity", data_second=grav_est_norm, label2="Estimated gravity", playback_speed=2, savepath="lfilter.mp4", suptitle=f"Loss: {format(loss,'.3f')}")


def main():
    # output loss for different files
    paths = [
        "/Users/gustaf/Documents/skola/exjobb/tremor/data/orientation_test/apple_test.csv",
        "/Users/gustaf/Documents/skola/exjobb/tremor/data/orientation_test/flat.csv",
        "/Users/gustaf/Documents/skola/exjobb/tremor/data/orientation_test/TREMOR12_samples_2022_08_27_1110.csv",
        "/Users/gustaf/Documents/skola/exjobb/tremor/data/orientation_test/TREMOR12-samples-2022-08-27-1108.csv",
    ]
    losses = np.zeros((len(paths),))

    for i, path in enumerate(paths):
        apple_data = data_parsing.read_apple(path)
        apple_grav = apple_data.loc[:,["gravX", "gravY", "gravZ"]].to_numpy().T

        # add linear acceleration to the apple data
        apple_data_noised = apple_data.loc[:,["gravX", "gravY", "gravZ"]].to_numpy().T * 9.82 + apple_data.loc[:,["accX", "accY", "accZ"]].to_numpy().T
        # extract orientation from apple_data_noised
        grav_est = go.get_orientation_naive(apple_data_noised)
        grav_est_norm = np.array([item / np.sum(item**2)**0.5 for item in grav_est.T]).T

        
        loss = np.mean(np.sum(apple_grav - grav_est_norm, axis=0)**2)
        losses[i] = loss
    for loss in losses:
        print(format(loss,".5f"))

if __name__ == '__main__':
    main()

