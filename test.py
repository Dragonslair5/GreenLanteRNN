import numpy as np
from libs.make_sample import baseCorrelationResult, addNoise, makeDataset
from models.network import Rnn


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

if __name__ == "__main__":

    train_data_length = 50
    in_out_neurons = 100
    n_hidden = 10
    model = Rnn(length_of_sequence=100, in_out_neurons=in_out_neurons, n_hidden=n_hidden)
    model.load("model.h5", compile=False)

    """

    Make sample data

    noise_data = np.array(baseCorrelationResult())
    correct_data = np.array(addNoise(addNoise(noise_data, complexity=5, complex_weight=0.5, noise_amp=1), complexity=5, complex_weight=100, noise_amp=1))
    """

    ###
    ##
    #  write here code to load test data.
    ##
    ###

    predict_data = np.zeros([1, (len(noise_data[0]) // in_out_neurons) * in_out_neurons + in_out_neurons])
    for offset in range(0, len(noise_data[0]), in_out_neurons):
        chunked_noise_data = noise_data[0, offset : offset + in_out_neurons]
        if chunked_noise_data.shape[0] < in_out_neurons:
            chunked_noise_data = np.pad(chunked_noise_data, (0, in_out_neurons - chunked_noise_data.shape[0]), "constant")

        input_data = chunked_noise_data.reshape(1, in_out_neurons, 1)
        predict_chunk = model.predict(input_data)
        predict_data[0, offset : offset + in_out_neurons] = predict_chunk[0]
    predict_data = predict_data[:, :correct_data.shape[1]]

    correct_x = np.arange(2, correct_data.shape[1], 3)
    noise_x = np.arange(0, noise_data.shape[1], 3)
    predict_x = np.arange(1, predict_data.shape[1], 3)


    fig = plt.figure(figsize=(12.0, 3.0))
    plt.bar(correct_x, correct_data[0, :correct_x.shape[0]], label='Correct data', color='orange')
    plt.bar(noise_x, noise_data[0, :noise_x.shape[0]], label='Noise data', color='blue')
    plt.bar(predict_x, predict_data[0, :predict_x.shape[0]], label='Predict data', color='purple')
    plt.legend(["Correct", "Noise", "Predict"], prop={'size':16,})
    plt.xlabel("Index [a.u]")
    plt.ylabel("Amp [a.u]")
    plt.show()
    fig.savefig("./correct_noise_predict.png")
    plt.close()


    sub_data = np.abs(correct_data - predict_data)

    fig = plt.figure(figsize=(12.0, 3.0))
    plt.bar(np.arange(sub_data.shape[1]), sub_data[0], label='Correct - Predict', color='red')
    plt.legend(["Correct - Predict"], prop={'size':16,})
    plt.xlabel("Index [a.u]")
    plt.ylabel("Amp [a.u]")
    plt.show()
    fig.savefig("./error_correct-predict.png")
    plt.close()

    sub_data = np.abs(correct_data - noise_data)

    fig = plt.figure(figsize=(12.0, 3.0))
    plt.bar(np.arange(sub_data.shape[1]), sub_data[0], label='Correct - Predict', color='red')
    plt.legend(["Correct - Noise"], prop={'size':16,})
    plt.xlabel("Index [a.u]")
    plt.ylabel("Amp [a.u]")
    plt.show()
    fig.savefig("./error_correct-noise.png")
    plt.close()