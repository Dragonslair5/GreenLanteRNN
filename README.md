# Green Lantern Project

	1) Create Signal
- createTrainingSamples.sh
- peek2wave.py
- (folder) trainingSamples/

> bash createTrainingSamples.sh

Samples will be created on trainingSamples folder, both for perfect and noisy signal.

	2) Merge all signals

- aConcatener.sh
- (folder) trainingSamples/
> cd trainingSamples
> bash aConcatener.sh

All samples are merged into a single sample for each of the cases, resulting in only 1 perfect signal (__signal_peek.csv__) and 1 noisy signal (__signal_noise.csv__).

	3) Train

- train.py
- train.sh
- trainingSamples/signal_peek.csv
- trainingSamples/signal_noise.csv

Creation and training  of the RNN occur by utilizing the train.py script.

> python3 train.py [number of epochs]
> e.g: python3 train.py 10

A bash script is also provided to automatize the number of epochs to be tested (__train.sh__).

	4) Testing (TODO)

Script __test.py__ still underdevelopment.	