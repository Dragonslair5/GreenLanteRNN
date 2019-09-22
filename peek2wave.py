import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

import sys



def generateTrainData(peek_count=16, peek_max=140, sparsely=20, filter_width=100, filter_sigma=10, noise_amp_ratio=0.2, noise_shift=9):

    peek = np.random.rand(peek_count) * peek_max
    sparsed_peek = np.pad(peek.reshape(peek.shape[0], 1), [(0, 0), (sparsely, sparsely)], "constant").reshape(peek.shape[0] * (sparsely * 2 + 1))
    #sparsed_peek = np.pad(sparsed_peek, (sparsely, sparsely), "constant")

    gaussian_filter = np.exp(-np.abs(np.arange(-filter_width, filter_width)) ** 2 / filter_sigma)
    noisy_gaussian_filter = gaussian_filter + noise_amp_ratio * np.exp(-(np.arange(-filter_width - noise_shift, filter_width - noise_shift)) ** 2 / filter_sigma)
    
    conv_peek = np.convolve(gaussian_filter, sparsed_peek, mode="same")
    noise_peek = np.convolve(noisy_gaussian_filter, sparsed_peek, mode="same")

    return conv_peek, noise_peek

if __name__ == "__main__":
    conv_peek, noise_peek = generateTrainData()

    fig = plt.figure(figsize=(10.0, 3.0))
    plt.plot(np.arange(0, conv_peek.shape[0]), conv_peek, label='Conv', color='green')
    plt.plot(np.arange(0, noise_peek.shape[0]), noise_peek, label='Conv', color='purple')
    plt.legend(["Gauss conv", "Noise conv"], prop={'size':16,})
    plt.xlabel("Degree [deg]")
    plt.ylabel("Amp [a.u]")
    plt.show()
    fig.savefig("./peek.png")
    plt.close()

    testNumber = sys.argv[1]

    np.savetxt("peek" + "%03d"%int(testNumber) + ".csv", X=conv_peek.reshape(conv_peek.shape[0], 1))
    np.savetxt("noise" + "%03d"%int(testNumber) + ".csv", X=noise_peek.reshape(noise_peek.shape[0], 1))
    

    
