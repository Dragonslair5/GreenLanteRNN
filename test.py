import numpy as np
import peek2wave as p2w
from models.network import Rnn
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

if __name__ == "__main__":

    train_data_length = 50
    in_out_neurons = 100
    n_hidden = 10
    model_name = "model.h5"
    print(sys.argv)
    if len(sys.argv) < 2:
        print("Usage: python test.py model_path")
        exit()
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    model = Rnn(length_of_sequence=100, in_out_neurons=in_out_neurons, n_hidden=n_hidden, model_load=True)
    model.load(model_name, compile=False)

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

    maximum_denoise = -100000

    for test_times in range(0, 100):

        correct_wave, noise_wave = p2w.generateTrainData()

        wave_length = correct_wave.shape[0]
        predict_wave = np.array([])
        predict_length = 100
        for offset in range(0, wave_length, predict_length):
            if wave_length - offset < predict_length:
                input_wave = noise_wave[offset : wave_length]
                input_wave = np.pad(input_wave, (0, predict_length - wave_length + offset), "constant")
                predict_chunk_wave = model.predict(input_wave.reshape(1, predict_length, 1))
                predict_wave = np.hstack((predict_wave, predict_chunk_wave[0, 0 : wave_length - offset]))
            else:
                input_wave = noise_wave[offset : offset + predict_length]
                predict_chunk_wave = model.predict(input_wave.reshape(1, predict_length, 1))
                predict_wave = np.hstack((predict_wave, predict_chunk_wave[0, 0 : predict_length]))

        input_noise_power = np.average(np.abs(noise_wave - correct_wave))
        predict_noise_power = np.average(np.abs(predict_wave - correct_wave))

        if maximum_denoise < (input_noise_power - predict_noise_power):
            best_correct_wave = correct_wave
            best_input_wave = noise_wave
            best_predict_wave = predict_wave
            print("{}: {} -> {} ({} -> {})".format(test_times, input_noise_power, predict_noise_power, maximum_denoise, input_noise_power - predict_noise_power))
            maximum_denoise = input_noise_power - predict_noise_power
        else:
            print("{}: {} -> {} ({})".format(test_times, input_noise_power, predict_noise_power, input_noise_power - predict_noise_power))


    fig = plt.figure(figsize=(10.0, 3.0))
    plt.plot(np.arange(0, best_correct_wave.shape[0]), best_correct_wave, label='Conv', color='green')
    plt.plot(np.arange(0, best_input_wave.shape[0]), best_input_wave, label='Conv', color='purple')
    plt.legend(["Gauss conv", "Noise conv"], prop={'size':16,})
    plt.xlabel("Degree [deg]")
    plt.ylabel("Amp [a.u]")
    plt.show()
    fig.savefig("./peek.png")
    plt.close()

    graph_x = np.arange(best_correct_wave.shape[0])

    fig = plt.figure(figsize=(12.0, 3.0))
    plt.plot(graph_x, best_correct_wave, label='Correct Sin wave', color='red')
    plt.plot(graph_x, best_input_wave, label='Input Sin wave', color='green')
    plt.plot(graph_x, best_predict_wave, label='Predict Sin wave', color='purple')
    plt.legend(["Correct Sin wave", "Input Sin wave", "Predict Sin wave"], prop={'size':16,})
    plt.xlabel("Degree [deg]")
    plt.ylabel("Amp [a.u]")
    plt.show()
    fig.savefig("./peek-predict.png")
    plt.close()

    fig = plt.figure(figsize=(12.0, 3.0))
    plt.plot(graph_x, best_input_wave, label='Input Sin wave', color='green')
    plt.legend(["Input Sin wave"], prop={'size':16,})
    plt.xlabel("Degree [deg]")
    plt.ylabel("Amp [a.u]")
    plt.show()
    fig.savefig("./noise.png")
    plt.close()


    fig = plt.figure(figsize=(12.0, 3.0))
    plt.plot(graph_x, best_predict_wave, label='Predict Sin wave', color='purple')
    plt.legend(["Predict Sin wave"], prop={'size':16,})
    plt.xlabel("Degree [deg]")
    plt.ylabel("Amp [a.u]")
    plt.show()
    fig.savefig("./predict.png")
    plt.close()


    fig = plt.figure(figsize=(12.0, 3.0))
    plt.plot(graph_x, best_correct_wave, label='Correct wave', color='red')
    plt.legend(["Correct wave"], prop={'size':16,})
    plt.xlabel("Degree [deg]")
    plt.ylabel("Amp [a.u]")
    plt.show()
    fig.savefig("./correct.png")
    plt.close()

    distanceFromCorrectAnswer = best_correct_wave - best_predict_wave
    originalDistanceFromCorrectAnswer= best_correct_wave - best_input_wave

    
    fig = plt.figure(figsize=(12.0, 3.0))
    plt.plot(graph_x, originalDistanceFromCorrectAnswer, label='Original distance from correct answer', color='red')
    plt.legend(["Original distance from correct answer"], prop={'size':16,})
    
    plt.xlabel("Degree [deg]")
    plt.ylabel("Amp [a.u]")
    axesA = plt.gca()
    axesA.set_ylim([-25,25])
    plt.show()
    fig.savefig("./original_distance_from_correct_answer.png")
    plt.close()

    fig = plt.figure(figsize=(12.0, 3.0))
    plt.plot(graph_x, distanceFromCorrectAnswer, label='Distance from correct answer', color='red')
    plt.legend(["Distance from correct answer"], prop={'size':16,})
    plt.xlabel("Degree [deg]")
    plt.ylabel("Amp [a.u]")
    axesB = plt.gca()
    axesB.set_ylim([-25,25])
    plt.show()
    fig.savefig("./distance_from_correct_answer.png")
    plt.close()

