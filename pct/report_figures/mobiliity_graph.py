import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import interpolate
import scipy.signal as signal

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    xs = np.linspace(0,2,200)
    y1 = 0.1 + 0.1*np.sin(xs[:20])

    b = 2*np.log(0.2) - np.log(0.8)
    a = 5*np.log(0.8/0.2)

    print(f"a: {a}")
    s = 0.5* sigmoid(np.linspace(-8,8,20))
    y2 = y1[-1] - s[0] + s

    e = np.exp(-np.linspace(0.8,5,160))
    y3 = y2[-1] - e[0] + e
    y = np.concatenate((y1,y2,y3))
    yhat = signal.filtfilt([1/3]*3, [1], y)
    
    y2 = np.sin(xs*2)/20 + 0.3
    plt.plot(xs, yhat, label="Mobility for PD patient")
    plt.plot(xs, y2, label="Mobility for reference person")
    plt.xlim([0,2])
    plt.ylim([0,0.7]) 

    plt.scatter(xs[20], yhat[20], marker="x", s=70, label="Medication taken")
    plt.tick_params(left = False, labelleft=False, labelbottom=False, bottom=False)
    plt.ylabel("Mobility")
    plt.xlabel("Time")
    plt.axhline(0.4, linestyle="dashed", color="red", label="Good mobility window")
    plt.axhline(0.2, linestyle="dashed", color="red")
    plt.grid(True, color="#93a1a1", alpha=0.3)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()

