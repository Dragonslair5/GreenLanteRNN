
myZero=0

rm signal_noise.csv
for i in noise*.csv
do
    cat $i >> signal_noise.csv
    # Padding 0s by the end of the signal
    # Resulting in 700 points per signal sample
    for i in $(seq 0 43)
    do
        echo -e $myZero >> signal_noise.csv
    done
done

rm signal_peek.csv
for i in peek*.csv
do
    cat $i >> signal_peek.csv
    # Padding 0s by the end of the signal
    # Resulting in 700 points per signal sample
    for i in $(seq 0 43)
    do
        echo -e $myZero >> signal_peek.csv
    done
done