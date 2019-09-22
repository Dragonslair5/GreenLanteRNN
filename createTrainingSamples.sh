
nSamples=100

cd trainingSamples
pwd
for (( i=1; i <= $nSamples; i=i+1 ))
do
    #x=10
    python ../peek2wave.py $i
done
cd ..
pwd