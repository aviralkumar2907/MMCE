#!/bin/bash

####### Experiment 1: Get ECE, Brier Scores, test NLL output for 5 different runs
echo "Experiment 1 --------------------------------------------"
for i in `seq 1 1`;
do
	echo 'Run #' $i
	python 20ng_mmce.py --mmce_coeff=10.0 > outfile.txt
	python get_calibration.py --file_name="outfile.txt" --T=1.0 --N=10 --num_classes=20
done

######## Experiment 2: Get trends with varying lambda values ########
echo "Experiment 2 --------------------------------------------"
for i in 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.00 12.0 13.0
do
	echo 'Lambda' $i
	python 20ng_mmce.py --mmce_coeff=$i > outfile.txt
	python get_calibration.py --file_name="outfile.txt" --T=1.0 --N=10 --num_classes=20 --visualize=False
done
