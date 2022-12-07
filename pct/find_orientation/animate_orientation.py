
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

import pct.preprocessing.data_parsing as dp

import pandas as pd
import numpy as np

def animate_orientation(data_first, label = "first data", data_second=None, label2 = "second data", data_third=None, label3 = "third data", step_size=10, savepath=None, limits=None, playback_speed=1, suptitle=""):
    # Time axis is axis 1
    assert data_first.shape[0] == 3, "First dimension has to be of size 3, this is the coordinates"
    if data_second is not None: assert data_second.shape[0], "First dimension has to be of size 3, these are the coordinates"
    global limits_used
    limits_used = []
    if limits is None:
        limits_used = [-1,1]
    else:
        limits_used = limits

    U, V, W = zip(*data_first.T)
    U2 = V2 = W2 = None
    U3 = V3 = W3 = None

    if data_second is not None:
        U2, V2, W2 = zip(*data_second.T)
    
    if data_third is not None:
        U3, V3, W3 = zip(*data_third.T)

    def update(num):
        ax.set_title(f"frame {str(num)}")
        u = U[num]
        v = V[num]
        w = W[num]
        new_segs = [[[0,0,0],[u,v,w]]]
        quiver.set_segments(new_segs)
        if data_second is not None:
            update2(num)
        if data_third is not None:
            update3(num)

    def update2(num):
        u = U2[num]
        v = V2[num]
        w = W2[num]
        new_segs = [[[0,0,0],[u,v,w]]]
        quiver2.set_segments(new_segs)

    def update3(num):
        u = U3[num]
        v = V3[num]
        w = W3[num]
        new_segs = [[[0,0,0],[u,v,w]]]
        quiver3.set_segments(new_segs)
    
    def onpress(event):
        global limits_used
        if event.key == "Q": exit()
        elif event.key == " ":
            ani.pause()
        elif event.key == "c":
            ani.resume()
        elif event.key == "+":
            limits_used = [limits_used[0] * 0.7, limits_used[1] * 0.7]
            set_lim(ax, limits_used)
        elif event.key == "-":
            limits_used = [limits_used[0]/0.7, limits_used[1]/0.7]
            set_lim(ax, limits_used)

    fig = plt.figure()
    fig.suptitle(suptitle)
    fig.canvas.mpl_connect("key_press_event", onpress)
    ax = fig.add_subplot(projection='3d')

    
    
   

    def set_lim(ax, lim):
        ax.set_xlim3d(lim)
        ax.set_ylim3d(lim)
        ax.set_zlim3d(lim)

    set_lim(ax, limits_used)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    quiver = ax.quiver(0, U[0], 0, V[0], 0, W[0], label=label)
    

    quiver2 = None
    if data_second is not None:
        quiver2 = ax.quiver(0, U2[0], 0, V2[0], 0, W2[0], color="red", label=label2)
    
    quiver3 = None
    if data_third is not None:
        quiver3 = ax.quiver(0, U3[0], 0, V3[0], 0, W3[0], color="green", label=label3)
    
    ax.legend()
    step_size = 5 # time steps per frame
    delay = 1000 * step_size // (100 * playback_speed) # delay in ms

    
    ani = animation.FuncAnimation(fig, update, range(0,len(U), step_size), interval=delay, blit=False)
    print(dir(ani))
    if savepath is not None:
        assert savepath.endswith(".mp4"), "Savpath must end with '.mp4'."
        ani.save(savepath, writer='ffmpeg')
    else : plt.show()


def main():
    path = "/Users/gustaf/Downloads/TREMOR12_samples_2022_11_29_1134.csv"
    df = pd.read_csv(path)

    df = df[["accX", "accY", "accZ", "gravX", "gravY", "gravZ"]]
    df.rename(columns= {"accX": "linaccX", "accY": "linaccY", "accZ": "linaccZ"}, inplace=True)
    data = df[["gravX", "gravY", "gravZ"]].to_numpy().T
    
    df = dp.read_raw("/Users/gustaf/Documents/skola/exjobb/tremor/data/raw_all/trc_2177.txt")
    accs_raw = np.array([
        data["accX"],
        data["accY"],
        data["accZ"],
    ]).reshape((3,-1))
    grav = get_orientation_naive(accs_raw)
    print(df)
    exit()
    

    animate_orientation(data, label = "first data", step_size=10)


if __name__ == '__main__':
    main()

