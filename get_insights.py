import numpy as np 
import sys
import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('file_name', None,
                    'Name of the output file generated from the model.')
flags.DEFINE_string('baseline_file_name', None,
                    'Name of the output file with baseline results.')
flags.DEFINE_float('T', 1.0, 'Temperature used to scale logits')
flags.DEFINE_integer('N', 10, 'Number of bins to be used for ECE measurement.')
flags.DEFINE_integer('num_classes', 20,
                     'Number of output classes for the task.')

FLAGS = flags.FLAGS

file_name = FLAGS.file_name
M = 3000
N = FLAGS.N
T = FLAGS.T
baseline_file_name = FLAGS.baseline_file_name


def get_bin_num(val):
  global N
  val = min(int(val*N), N)
  return val

def softmax(arr):
  arr1 = arr - np.expand_dims(np.max(arr, 1), 1)
  return np.exp(arr1)/np.expand_dims(np.sum(np.exp(arr1), 1), 1)


mmce_list = []
correct_list = []
correct_list_baseline = []
baseline_list = []
mmce_prob_values = []
baseline_prob_values = []
baseline_temp_prob_values = []

with open(baseline_file_name, 'r') as g:
  with open(file_name, 'r') as f:
    line = f.readline()
    line2 = g.readline()
    while line or line2:    
      def get_data_from_file(line, f):
        pos = line.index('Targets:')
        line = line[pos+10:]
        line = line.replace('][', ' ')
        line = line.replace(']', '')
        line = line.replace('[', '')
        line = line.replace('\n', '')
        line = line.replace(',', '')
        line = line.split(' ')
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

        return xent_index, probs, predictions

      if 'Targets' in line:
        tgt1, probs1, _ = get_data_from_file(line, f)
        mmce_list.extend(np.argmax(probs1, 1).tolist())
        mmce_prob_values.extend(np.max(probs1, 1).tolist())
        correct_list.extend(tgt1.tolist())
        
      if 'Targets' in line2:
        tgt2, probs2, _ = get_data_from_file(line2, g)
        baseline_list.extend((np.argmax(probs2, 1)).tolist())
        baseline_prob_values.extend(np.max(probs2, 1).tolist())
        baseline_temp_prob_values.extend(
        	np.max(softmax(np.log(probs2 + 1e-10)*T), 1).tolist())
        correct_list_baseline.extend(tgt2.tolist())

      line = f.readline()
      line2 = g.readline()

count_high_baseline = np.sum(np.array(baseline_prob_values) >=0.99)
count_high_baseline_temp = np.sum(np.array(baseline_temp_prob_values) >= 0.99)
count_high_mmce = np.sum(np.array(mmce_prob_values) >= 0.99)
print ('Baseline: ' , 1.0*count_high_baseline/len(baseline_prob_values))
print ('Baseline + T: ', 1.0*count_high_baseline_temp/len(baseline_prob_values))
print ('MMCE: ', 1.0*count_high_mmce/len(baseline_prob_values))


def get_acc(prob_list, tgt_list, pred_list):
    avg_accuracy = ((np.array(prob_list) >= 0.99)*\
    		(np.array(tgt_list) == np.array(pred_list)))
    return np.sum(avg_accuracy)

baseline_high_acc = 1.0*get_acc(baseline_prob_values,
	correct_list_baseline, baseline_list)/count_high_baseline
baseline_temp_high_acc = 1.0*get_acc(baseline_temp_prob_values,
	correct_list_baseline, baseline_list)/count_high_baseline_temp
mmce_high_acc = 1.0*get_acc(mmce_prob_values, 
																			correct_list, mmce_list)/count_high_mmce

print ('Baseline: ', baseline_high_acc)
print ('Baseline + T: ', baseline_temp_high_acc)
print ('MMCE: ', mmce_high_acc)

