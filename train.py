import numpy as np
#from libs.make_sample import baseCorrelationResult, addNoise, makeDataset
from models.network import Rnn
import sys

if __name__ == "__main__":

    #chunk size
    chunkSize = 100 

    Tepoch= int(sys.argv[1])

    """

    Code: Make sample dataset

    noise_data = np.array([baseCorrelationResult() for i in range(train_data_length)])
    correct_data = np.array([addNoise(addNoise(noise_data[i], complexity=5, complex_weight=0.5, noise_amp=1), complexity=5, complex_weight=100, noise_amp=1) for i in range(train_data_length)])

    train_data, valid_data = makeDataset(noise_data, correct_data, train_data_length=train_data_length, length_of_sequence=100, max_index=1024)
    """

    ###
    ##
    #  write here ready to train data and correct data code
    ##
    ###

    perfectSignal=np.loadtxt("trainingSamples/signal_peek.csv")
    noisySignal=np.loadtxt("trainingSamples/signal_noise.csv")
    
    #print(len(perfectSignal))


    perfectSignalSamples=[]
    noisySignalSamples=[]
    completeSamples=int(len(perfectSignal)/chunkSize)
    remainderSampleLength=len(perfectSignal)%100

    print("remainderSampleLength="+str(remainderSampleLength))

    print("completeSamples="+str(completeSamples))

    for i in range(0,completeSamples):
        #print("aaa" + str(int(i*chunkSize)))
        perfectSignalSamples.append(perfectSignal[int(i*chunkSize):int((i+1)*chunkSize):1])
        noisySignalSamples.append(noisySignal[int(i*chunkSize):int((i+1)*chunkSize):1])


    print("size = "+str(len(perfectSignalSamples)))


    if remainderSampleLength != 0:
        lastChunk = perfectSignal[int(completeSamples*100):int(completeSamples*100 + remainderSampleLength):1]
        lastNoisyChunk = noisySignal[int(completeSamples*100):int(completeSamples*100 + remainderSampleLength):1]
        for i in range(0, 100-remainderSampleLength):
            lastChunk= np.append(lastChunk, 0)
            lastNoisyChunk= np.append(lastNoisyChunk, 0)

        print(lastChunk)

        perfectSignalSamples= np.append(perfectSignalSamples,lastChunk)
        noisySignalSamples= np.append(noisySignalSamples,lastNoisyChunk)
   # else:
        


    perfectSignalSamples = np.array(perfectSignalSamples)
    noisySignalSamples = np.array(noisySignalSamples)
    
    print("size=" + str(len(perfectSignalSamples)))


    #print(perfectSignalSamples.reshape(len(perfectSignalSamples), chunkSize, 1))
    perfectSignalSamples = perfectSignalSamples.reshape(perfectSignalSamples.shape[0], perfectSignalSamples.shape[1])
    print("########")
    noisySignalSamples = noisySignalSamples.reshape(noisySignalSamples.shape[0], noisySignalSamples.shape[1], 1)


   # exit(0)

    #for x in range(1, 30):
    #for(i=0; i < 100; i++)

	#chunkNoisy[]
    #	chunkSignal[]

    # length of RNN output result
    ## maybe it is equal to length_of_sequence
    in_out_neurons = 100

    # hidden layers' count
    ## count higher, networks will be more complex, but also networks needs more power to train.
    n_hidden = 300

    model = Rnn(length_of_sequence=chunkSize, in_out_neurons=in_out_neurons, n_hidden=n_hidden, epoch=Tepoch)
    model.train(noisySignalSamples, perfectSignalSamples)

    # networks will be output as model.h5 at current working directory
    model.save('model.h5', include_optimizer=False)
