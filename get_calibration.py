'''Script to get the reliability diagrams, expected calibration error,
   Brier score, test-NLL value for a dataset.'''

import numpy as np 
import sys
import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf 

flags = tf.app.flags
flags.DEFINE_string('file_name', None,
                    'Name of the output file generated from the model.')
flags.DEFINE_float('T', 1.0, 'Temperature used to scale logits')
flags.DEFINE_integer('N', 10, 'Number of bins to be used for ECE measurement.')
flags.DEFINE_integer('num_classes', 20,
                     'Number of output classes for the task.')
flags.DEFINE_bool('visualize', False,
                  'Whether we want to visualise or not.')

FLAGS = flags.FLAGS

matplotlib.use('GTK')
plt.style.use('ggplot')

file_name = FLAGS.file_name
M = 3000
N = FLAGS.N
T = FLAGS.T

cnt = 0.0
num_calls = 0.0

# Arrays to store the binwise counts of correct and observed predictions.
bin_array = np.zeros((N+1,))
means_array = np.zeros((N+1,))
num_array = np.zeros((N+1,)) 
val_out = 0
val_incorrect = 0

def get_bin_num(val):
  global N
  val = min(int(val*N), N)
  return val

def softmax(arr):
  arr1 = arr - np.max(arr)
  return np.exp(arr1)/np.sum(np.exp(arr1))

brier_score = 0.0
total_num = 0.0
nll = 0.0

# Parsing specific to this codebase. Can be changed based on need.
with open(file_name, 'r') as f:
  line = f.readline()
  while line:
    if 'Targets' in line:
      pos = line.index('Targets:')
      line = line[pos+10:]
      line = line.replace('][', ' ')
      line = line.replace(']', '')
      line = line.replace('[', '')
      line = line.replace('\n', '')
      line = line.replace(',', '')
      line = line.split(' ')
      # print (line)
      line = [int(x) for x in line]
      xent_index = np.array(line)

      line = f.readline()
      line = f.readline()
      pos = line.index('Probs:')
      line = line[pos+8:]
      line = line.replace('][', ' ')
      line = line.replace(']', '')
      line = line.replace('[', '')
      line = line.replace('\n', '')
      line = line.replace(',', '')
      line = line.split(' ')
      line = [float(x) for x in line]

      line = np.array(line)
      probs = np.reshape(line, (-1, FLAGS.num_classes))
      predictions = np.argmax(probs, axis=1)

      for i in range(xent_index.shape[0]):
        target = xent_index[i]
        dist = softmax(np.log(probs[i]+1e-5)*T)
        pred = predictions[i]

        ttz = np.zeros(20)
        ttz[target] = 1.0
        brier_score += np.sum((dist - ttz)**2)
        total_num += 1.0

        nll -= np.log(dist)[target]

        bin_num = get_bin_num(dist[pred])
        means_array[bin_num] += dist[pred]
        num_array[bin_num] += 1.0
        if pred == target:
            bin_array[bin_num] += 1.0
    line = f.readline()

print ('Brier score: ', brier_score/total_num)
print ('NLL: ', nll/total_num)
means_array[N-1] += means_array[N]
bin_array[N-1] += bin_array[N]
num_array[N-1] += num_array[N]
means_array = means_array/(num_array+1e-5)
bin_array = bin_array/(num_array+1e-5)

ece = np.abs(bin_array - means_array)
ece = ece*num_array
ece = np.sum(ece[:-1])/np.sum(num_array[:-1])
print ('Expected Calibration Error: ', ece)

if FLAGS.visualize:
  plt.figure()
  plt.plot(means_array[:-1], bin_array[:-1], linewidth=3.0)
  plt.plot(0.1*np.arange(11), 0.1*np.arange(11), linewidth=3.0)
  plt.plot(means_array[:-1], num_array[:-1]/np.sum(num_array[:-1]))
  plt.scatter(means_array[:-1], num_array[:-1]/np.sum(num_array[:-1]))
  plt.show()
