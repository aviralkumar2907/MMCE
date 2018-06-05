from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_float('mmce_coeff', 4.0,
                   'Coefficient for MMCE error term.')
flags.DEFINE_integer('batch_size', 128, 'Batch size for training.')
flags.DEFINE_integer('num_epochs', 22, 'Number of epochs of training.')
FLAGS = flags.FLAGS

BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                if sys.version_info < (3,):
                    f = open(fpath)
                else:
                    f = open(fpath, encoding='latin-1')
                t = f.read()
                i = t.find('\n\n')  # skip header
                if 0 < i:
                    t = t[i:]
                texts.append(t)
                f.close()
                labels.append(label_id)

print('Found %s texts.' % len(texts))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-(num_validation_samples+900)]
y_train = labels[:-(num_validation_samples+900)]
x_pval = data[data.shape[0]-num_validation_samples-900:(data.shape[0]
                                                    -num_validation_samples)]
y_pval = labels[data.shape[0]-(num_validation_samples+900):(data.shape[0]
                                                    -num_validation_samples)]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print (data.shape[0] - num_validation_samples)
print ('XPVAL: ', x_pval.shape, data.shape)

print('Preparing embedding matrix.', x_train.shape)

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

def get_out_tensor(tensor1, tensor2):
  return tf.reduce_mean(tensor1*tensor2)

def calibration_unbiased_loss(logits, correct_labels):
  """Function to compute MMCE_m loss."""  
  predicted_probs = tf.nn.softmax(logits)
  pred_labels = tf.argmax(predicted_probs, 1)
  predicted_probs = tf.reduce_max(predicted_probs, 1)
  correct_mask = tf.where(tf.equal(pred_labels, correct_labels),
                          tf.ones(tf.shape(pred_labels)),
                          tf.zeros(tf.shape(pred_labels)))
  c_minus_r = tf.to_float(correct_mask) - predicted_probs
  dot_product = tf.matmul(tf.expand_dims(c_minus_r, 1),
                          tf.transpose(tf.expand_dims(c_minus_r, 1)))
  tensor1 = predicted_probs
  prob_tiled = tf.expand_dims(tf.tile(tf.expand_dims(tensor1, 1),
                              [1, tf.shape(tensor1)[0]]), 2)
  prob_pairs = tf.concat([prob_tiled, tf.transpose(prob_tiled, [1, 0, 2])],
                         axis=2)

  def tf_kernel(matrix):
    return tf.exp(-1.0*tf.abs(matrix[:, :, 0] - matrix[:, :, 1])/(2*0.2))  

  kernel_prob_pairs = tf_kernel(prob_pairs)
  numerator = dot_product*kernel_prob_pairs
  return tf.reduce_sum(numerator)/tf.square(tf.to_float(tf.shape(correct_mask)[0]))
        
def self_entropy(logits):
  probs = tf.nn.softmax(logits)
  log_logits = tf.log(probs + 1e-10)
  logits_log_logits = probs*log_logits
  return -tf.reduce_mean(logits_log_logits)

def calibration_mmce_w_loss(logits, correct_labels):
  """Function to compute the MMCE_w loss."""
  predicted_probs = tf.nn.softmax(logits)
  range_index = tf.to_int64(tf.expand_dims(tf.range(0,
                                            tf.shape(predicted_probs)[0]), 1))
  predicted_labels = tf.argmax(predicted_probs, axis=1)
  gather_index = tf.concat([range_index,
                            tf.expand_dims(predicted_labels, 1)], axis=1)
  predicted_probs = tf.reduce_max(predicted_probs, 1)
  correct_mask = tf.where(tf.equal(correct_labels, predicted_labels),
                          tf.ones(tf.shape(correct_labels)),
                          tf.zeros(tf.shape(correct_labels)))
  sigma = 0.2
  
  def tf_kernel(matrix):
    """Kernel was taken to be a laplacian kernel with sigma = 0.4."""
    return tf.exp(-1.0*tf.abs(matrix[:, :, 0] - matrix[:, :, 1])/(2*0.2))  

  k = tf.to_int32(tf.reduce_sum(correct_mask))
  k_p = tf.to_int32(tf.reduce_sum(1.0 - correct_mask))
  cond_k = tf.where(tf.equal(k, 0), 0, 1)
  cond_k_p = tf.where(tf.equal(k_p, 0), 0, 1)
  k = tf.maximum(k, 1)*cond_k*cond_k_p + (1 - cond_k*cond_k_p)*2 
  k_p = tf.maximum(k_p, 1)*cond_k_p*cond_k + ((1 - cond_k_p*cond_k)*
                                            (tf.shape(correct_mask)[0] - 2))
  correct_prob, _ = tf.nn.top_k(predicted_probs*correct_mask, k)
  incorrect_prob, _ = tf.nn.top_k(predicted_probs*(1 - correct_mask), k_p)
  
  def get_pairs(tensor1, tensor2):
    correct_prob_tiled = tf.expand_dims(tf.tile(tf.expand_dims(tensor1, 1),
                      [1, tf.shape(tensor1)[0]]), 2)
    incorrect_prob_tiled = tf.expand_dims(tf.tile(tf.expand_dims(tensor2, 1),
                      [1, tf.shape(tensor2)[0]]), 2)
    correct_prob_pairs = tf.concat([correct_prob_tiled,
                     tf.transpose(correct_prob_tiled, [1, 0, 2])],
                     axis=2)
    incorrect_prob_pairs = tf.concat([incorrect_prob_tiled,
                   tf.transpose(incorrect_prob_tiled, [1, 0, 2])],
                   axis=2)
    correct_prob_tiled_1 = tf.expand_dims(tf.tile(tf.expand_dims(tensor1, 1),
                        [1, tf.shape(tensor2)[0]]), 2)
    incorrect_prob_tiled_1 = tf.expand_dims(tf.tile(tf.expand_dims(tensor2, 1),
                        [1, tf.shape(tensor1)[0]]), 2)
    correct_incorrect_pairs = tf.concat([correct_prob_tiled_1,
                  tf.transpose(incorrect_prob_tiled_1, [1, 0, 2])],
                  axis=2)
    return correct_prob_pairs, incorrect_prob_pairs, correct_incorrect_pairs
  
  correct_prob_pairs, incorrect_prob_pairs,\
               correct_incorrect_pairs = get_pairs(correct_prob, incorrect_prob)
  correct_kernel = tf_kernel(correct_prob_pairs)
  incorrect_kernel = tf_kernel(incorrect_prob_pairs)
  correct_incorrect_kernel = tf_kernel(correct_incorrect_pairs)  
  sampling_weights_correct = tf.matmul(tf.expand_dims(1.0 - correct_prob, 1),
                           tf.transpose(tf.expand_dims(1.0 - correct_prob, 1)))
  correct_correct_vals = get_out_tensor(correct_kernel,
                                                    sampling_weights_correct)
  sampling_weights_incorrect = tf.matmul(tf.expand_dims(incorrect_prob, 1),
                           tf.transpose(tf.expand_dims(incorrect_prob, 1)))
  incorrect_incorrect_vals = get_out_tensor(incorrect_kernel,
                                                    sampling_weights_incorrect)
  sampling_correct_incorrect = tf.matmul(tf.expand_dims(1.0 - correct_prob, 1),
                           tf.transpose(tf.expand_dims(incorrect_prob, 1)))
  correct_incorrect_vals = get_out_tensor(correct_incorrect_kernel,
                                                    sampling_correct_incorrect)
  correct_denom = tf.reduce_sum(1.0 - correct_prob)
  incorrect_denom = tf.reduce_sum(incorrect_prob)
  m = tf.reduce_sum(correct_mask)
  n = tf.reduce_sum(1.0 - correct_mask)
  mmd_error = 1.0/(m*m + 1e-5) * tf.reduce_sum(correct_correct_vals) 
  mmd_error += 1.0/(n*n + 1e-5) * tf.reduce_sum(incorrect_incorrect_vals)
  mmd_error -= 2.0/(m*n + 1e-5) * tf.reduce_sum(correct_incorrect_vals)
  return tf.maximum(tf.stop_gradient(tf.to_float(cond_k*cond_k_p))*\
                                          tf.sqrt(mmd_error + 1e-10), 0.0)

def model(inputs, keep_prob):
    ''' Generate the CNN model '''
    W = tf.Variable(tf.zeros([num_words, EMBEDDING_DIM]),
                    trainable=False,
                    name='embed_weights')
    embedding_placeholder = tf.placeholder(tf.float32,
                                          [num_words, EMBEDDING_DIM])
    embedding_init = W.assign(embedding_placeholder)

    embedding_output = tf.nn.embedding_lookup(W, inputs)
    conv_1 = tf.layers.conv1d(embedding_output, 128, 5,  1,
                             'VALID', name='conv_layer1')
    conv_relu1 = tf.nn.relu(conv_1)
    pooled_1 = tf.layers.max_pooling1d(conv_relu1, 5, 1)
    conv_2 = tf.layers.conv1d(pooled_1, 128, 5,  1, 'VALID', name='conv_layer2')
    conv_relu2 = tf.nn.relu(conv_2)

    print ("Conv Relu3: ", conv_relu2)
    pooled_2 = tf.layers.max_pooling1d(conv_relu2, 5, 1)
    conv_3 = tf.layers.conv1d(pooled_2, 128, 5, 1, 'VALID', name='conv_layer3')
    conv_relu3 = tf.nn.relu(conv_3)

    # batch x step x feature_size
    # now global max pooling layer
    global_pool_out = tf.reduce_max(conv_relu3, axis=1)
    print ('Global pool out: ', global_pool_out)
    fc1 = tf.contrib.layers.fully_connected(global_pool_out, 128)
    fc1 = tf.nn.dropout(fc1, keep_prob)
    out_layer = tf.contrib.layers.fully_connected(fc1, 20,
                                                  activation_fn=tf.nn.softmax)

    return out_layer, embedding_init, embedding_placeholder

def add_loss(logits, true_labels):
    mmce_error = 1.0*calibration_mmce_w_loss(tf.log(logits + 1e-10), true_labels)
    ce_error = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.log(logits+1e-10),
                                                     labels=true_labels))
    return ce_error + FLAGS.mmce_coeff*mmce_error

def optimize(loss):
    opt = tf.train.AdamOptimizer()
    train_opt = opt.minimize(loss)
    return train_opt

input_placeholder = tf.placeholder(tf.int32, [None, None])
input_labels = tf.placeholder(tf.int64, [None])
keep_prob = tf.placeholder(tf.float32)

logits_layer, embedding_init,\
            embedding_placeholder = model(input_placeholder, keep_prob)
loss_layer = add_loss(logits_layer, input_labels)
train_op = optimize(loss_layer)

predictions = tf.argmax(logits_layer, 1)
acc = tf.reduce_sum(tf.where(tf.equal(predictions, input_labels),
                    tf.ones(tf.shape(predictions)),
                    tf.zeros(tf.shape(predictions))))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
sess.run(embedding_init, feed_dict={embedding_placeholder: embedding_matrix})

batch_size = FLAGS.batch_size
num_epochs = FLAGS.num_epochs

for epoch in range(num_epochs):
    perm = np.random.permutation(np.arange(len(x_train)))
    permutation_train = np.take(x_train, perm, axis=0)
    permutation_labels = np.take(y_train, perm, axis=0)

    overall_avg_loss = 0.0
    overall_acc = 0.0
    for i in range(permutation_train.shape[0]//batch_size):
        feed_dict = dict()
        feed_dict[input_placeholder] = permutation_train[i*batch_size:\
                                                             (i+1)*batch_size]
        feed_dict[input_labels] = np.argmax(permutation_labels[i*batch_size:\
                                                           (i+1)*batch_size], 1)
        feed_dict[keep_prob] = 0.7
        loss, _, acc_train = sess.run([loss_layer, train_op, acc],
                                      feed_dict=feed_dict)
        overall_avg_loss += loss
        overall_acc += acc_train
        
    print ('Train acc: ', overall_acc/permutation_train.shape[0])
    print ('Train Loss: ', overall_avg_loss)

    feed_dict = dict()
    print (x_pval.shape)
    feed_dict[input_placeholder] = x_pval
    feed_dict[input_labels] = np.argmax(y_pval, 1)
    feed_dict[keep_prob] = 1.0
    accuracy, val_loss = sess.run([acc, loss_layer], feed_dict=feed_dict)
    print ('Val accuracy: ', accuracy/x_pval.shape[0], val_loss)
    
    feed_dict = dict()
    feed_dict[input_placeholder] = x_val
    feed_dict[input_labels] = np.argmax(y_val, 1)
    feed_dict[keep_prob] = 1.0
    accuracy, loss = sess.run([acc, loss_layer], feed_dict=feed_dict)
    preds_t = sess.run(logits_layer, feed_dict=feed_dict)
    
    #print ('Targets: ', np.argmax(y_val, 1).tolist())
    #print ('Predictions: ', np.argmax(preds_t, 1).tolist())
    #print ('Probs: ', preds_t.tolist())

    print ('Accuracy, Loss: ', accuracy/x_val.shape[0], loss)

# Final testing after training, also print the targets and logits
# for computing calibration.
feed_dict = dict()
feed_dict[input_placeholder] = x_pval
feed_dict[input_labels] = np.argmax(y_pval, 1)
feed_dict[keep_prob] = 1.0
accuracy, logits = sess.run([acc, logits_layer], feed_dict=feed_dict)
print ('Targets: ', np.argmax(y_pval, 1).tolist())
print ('Predictions: ', np.argmax(logits, 1).tolist())
print ('Probs: ', logits.tolist())



