
nSamples=10


for (( i=1; i <= $nSamples; i=i+1 ))
do
    epoch=$(($i*100));
    echo $epoch
    python3 train.py $epoch > output.txt
    touch model.h5
    touch output.txt
    printf -v j "%04d" $epoch
    mv model.h5 Resultados/step_${j}.h5
    mv output.txt Resultados/step_${j}_output.txt
done
