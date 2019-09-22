import numpy as np
from numpy import random

def baseCorrelationResult(seed=None, min=0, max=10, max_index=1024):
    return random.rand(1, 1024) * (max + min) - min

def addNoise(result, complexity=5, complex_weight=5, noise_amp=10):
    if complexity == 0:
        return result
    noise_raw_sin = []
    for _ in range(complexity):
        random_noise = random.rand()
        noise_raw_sin.append(np.sin([
            3.1415 * k / 180 * random_noise * complex_weight - random_noise * complex_weight for k in range(0, result.shape[1])
        ]))
    noise_raw_sin = np.sum(noise_raw_sin, axis=0) / complexity * random.rand() * noise_amp
    return result + np.abs(noise_raw_sin)

def makeDataset(correct_data, noise_data, train_data_length=50, length_of_sequence=100, max_index=1024):
    train_data, valid_data = [], []

    for data_offset in range(train_data_length):
        noise_array = noise_data[data_offset]
        correct_array = noise_data[data_offset]
        for array_offset in range(0, max_index, length_of_sequence):
            cut_noise_array = noise_array[0, array_offset : array_offset + length_of_sequence]
            cut_correct_array = correct_array[0, array_offset : array_offset + length_of_sequence]

            # zero padding if cutted array's length is less than length of sequence
            if array_offset + length_of_sequence > max_index:
                zero_length = array_offset + length_of_sequence - max_index
                cut_noise_array = np.pad(cut_noise_array, (0, zero_length), "constant")
                cut_correct_array = np.pad(cut_correct_array, (0, zero_length), "constant")

            train_data.append(cut_noise_array)
            valid_data.append(cut_correct_array)

    reshaped_train_data = np.array(train_data).reshape(len(train_data), length_of_sequence, 1)
    reshaped_valid_data = np.array(valid_data).reshape(len(valid_data), length_of_sequence)

    return reshaped_train_data, reshaped_valid_data