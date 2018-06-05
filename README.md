# MMCE
This is an example of the Maximum Mean Calibration Error on 20 NewsGroups dataset with a global pooling CNN model, as described in the paper **Trainable Calibration Measures for Neural Networks from Kernel Mean Embeddings, ICML 2018**. The code is yet in its preliminary version and will be updated with other datasets, unit tests, comments and extensive documentation in near future. 

An example application of MMCE on 20 Newsgroups document classification is provided as of now. To run MMCE, follow the procedure given below:

In order to use MMCE for training, run:
```bash
python cnn_keras_preprocess_2.py --mmce_coeff=8.0 --batch_size=batch_size > outfile.txt
```
In order to compute the ECE numbers, Brier score and test-NLL or to visualize the reliability plots, we then run the ```get_calibration.py``` script.
```bash
python get_calibration.py --file_name=outfile.txt --T=1.0 --N=10 --num_classes=20 --visualize=True
```
The flag ```T``` is used to control the temperature at which calibration is to be measured. ```N``` is used to control the number of bins used to compute ECE and ```num_classes``` should be set to the number of possible output classes for the dataset. If ```visualize``` is set to True, a matplotlib based reliability diagram should be generated.

The code makes use of Glove embeddings, which need to be downloaded and kept. We expect that the glove embeddings should be in the directory ```glove.6B``` relative to the src file, and the dataset should be located in the ```20newsgroup``` directory inside the base directory. The code for pre-processing the documents is borrowed from the Keras Pre-trained word embedding tutorial at https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html. 

Given is a script **```run_mmce.sh```**, which contains code for running MMCE training, with a default ```batch size=128``` and ```mmce_coeff=8.0``` and then generate the ECE, Brier Score and test-NLL numbers. Also, it contains the experiment to run MMCE training for different values of ```mmce_coeff``` and then generates results on these metrics. In order to use the script, just run:
```bash
bash run_mmce.sh
```
## Dependencies
Tensorflow (>1.4), Keras (for preprocessing text files), Numpy, Python

## Results
We expect the ECE numbers to lie in the range of 6-7%, Brier score in the range of 0.35-0.38 and test NLL around 0.95-0.98 for the finetuned value of ```mmce_coeff=8.0```.  

## Disclaimer
This code is for educational purposes only. 
