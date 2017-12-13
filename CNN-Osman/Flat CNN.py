
# coding: utf-8

# In[1]:


import pandas
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
get_ipython().magic('matplotlib inline')


# ## Data Import

# In[2]:


path_train = 'resource/train.csv'
path_test = 'resource/test.csv'

train = pandas.read_csv(path_train)
test = pandas.read_csv(path_test)

train_corpus = train.as_matrix()[:,1]
test_corpus = test.as_matrix()[:,1]

#print(train_corpus)
print(train)
#print(test_corpus)
print(test)


# ## Data Preprocessing
# Split text into words, including punctuation.

# In[3]:


import re

def splitIntoWordList(text):
    return re.findall(r"\w+|[^\w\s]", text)

# Split the training text word by word, including punctuation as words 
train_text = list()
train_text_length = list()
for sentence in train['text']:
    l = splitIntoWordList(sentence)
    train_text.append(l)
    train_text_length.append(len(l))
# Split the testing text word by word, including punctuation as words
test_text = list()
test_text_length = list()
for sentence in test['text']:
    l = splitIntoWordList(sentence)
    test_text.append(l)
    test_text_length.append(len(l))

print(train_text[0])
print(len(train_text_length))    
print(test_text[0])
print(len(test_text_length))


# ## Data Visualization
# Find out how long sentences are and add to data

# In[4]:


train['length'] = train_text_length
test['length'] = test_text_length


# In[5]:


train['length'].describe()


# Set a maximum sentence length of 50 words. Sentence with less will be padded, any sentence with more will be truncated

# In[6]:


sequence_length = 100


# ## One Hot Encoding of Targets

# In[7]:


from keras.utils.np_utils import to_categorical

# One hot encoding of labels
train_labels = train.as_matrix()[:,2]
train_targets = np.zeros(train_labels.shape)
for idx, label in enumerate(train_labels):
    if 'EAP' == label:
        train_targets[idx] = 0
    elif 'HPL' == label:
        train_targets[idx] = 1
    elif 'MWS' == label:
        train_targets[idx] = 2
    else:
        raise ValueError("EAP, HPL or MWS is not in label")
    

# One-hot encode
encoded_train_targets = to_categorical(train_targets)

print(train_labels)
print(encoded_train_targets)


# ## Document Tokenization

# In[8]:


print(train_corpus)


# In[9]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

t = Tokenizer()
t.fit_on_texts(train_corpus)
# Break text into sequences and assign integer id to each word in sequence
encoded_train_text = t.texts_to_sequences(train_corpus)
# Pad text to a max length of sequence_length
padded_encoded_train_text = pad_sequences(encoded_train_text, maxlen=sequence_length, padding='post')

# Do the same for the test data
encoded_test_text = t.texts_to_sequences(test_corpus)
padded_encoded_test_text = pad_sequences(encoded_test_text, maxlen=sequence_length, padding='post')

print(padded_encoded_train_text.shape)
print(padded_encoded_test_text.shape)


# In[10]:


''' 
# Split data into training and validation sets
indices = np.arange(padded_encoded_train_text.shape[0])
# np.random.shuffle(indices)
shuffled_padded_encoded_train_text = padded_encoded_train_text[indices]
shuffled_encoded_train_targets = encoded_train_targets[indices]

VALIDATION_SPLIT = 0.2
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = shuffled_padded_encoded_train_text[:-nb_validation_samples]
y_train = shuffled_encoded_train_targets[:-nb_validation_samples]
x_val = shuffled_padded_encoded_train_text[-nb_validation_samples:]
y_val = shuffled_encoded_train_targets[-nb_validation_samples:]
'''
x_train = padded_encoded_train_text
y_train = encoded_train_targets


# ## Embedding Layer

# In[11]:


# Load the whole embedding into memory
embeddings_index = dict()

# f = open('resource/tr_data_embeddings.txt',encoding='utf8')
f = open('resource/glovew2v.txt',encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))


# In[12]:


# Dimensionality of each embedding
embedding_size = len(embeddings_index.get('the'))


# In[13]:


vocabulary_size = len(t.word_index) + 1
print(vocabulary_size)

# create a embedding matrix for words in training docs
embedding_matrix = np.zeros((vocabulary_size, embedding_size))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
from keras.layers import Embedding

# Input shape to embedding layer: 2D tensor - (batch_size, sequence_length)
# Output shape of embedding layer: 3D tensor - (batch_size, sequence_length, output_dim)
embedding_layer = Embedding(vocabulary_size,
                           embedding_size,
                           weights=[embedding_matrix],
                           input_length=sequence_length,
                           trainable=False)


# ## Convolutional Neural Network
# 
# ### Single Layer Multi Filter

# In[62]:


# EAP, HPL or MWS
num_classes= 3
# Filter Sizes
filter_sizes = [2,3]
# The number of filters for each filter size that runs through the whole sentence
num_filters = 250
batch_size = 64
num_epochs = 10


# ### Placeholders
# 
# The input images x will consist of a 2d tensor of floating point numbers. Here we assign it a shape of [None, 784], where 784 is the dimensionality of a single flattened 28 by 28 pixel MNIST image, and None indicates that the first dimension, corresponding to the batch size, can be of any size. The target output classes y_ will also consist of a 2d tensor, where each row is a one-hot 10-dimensional vector indicating which digit class (zero through nine) the corresponding MNIST image belongs to.
# 
# The shape argument to placeholder is optional, but it allows TensorFlow to automatically catch bugs stemming from inconsistent tensor shapes.

# In[63]:


input_x = tf.placeholder(tf.float32, shape=[None, sequence_length])
input_y = tf.placeholder(tf.float32, shape=[None, num_classes])
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")


# ### Weight Initialization
# To create this model, we're going to need to create a lot of weights and biases. One should generally initialize weights with a small amount of noise for symmetry breaking, and to prevent 0 gradients. Since we're using ReLU neurons, it is also good practice to initialize them with a slightly positive initial bias to avoid "dead neurons". Instead of doing this repeatedly while we build the model, let's create two handy functions to do it for us.

# ### Convolution and Pooling
# Our convolutions uses a stride of one and are zero padded so that the output is the same size as the input. Our pooling is plain old max pooling over 2x2 blocks.

# ### First Filter

# W is our pre-trained embedding matrix. embedding_layer() creates the actual embedding operation. The result of the embedding operation is a 3-dimensional tensor of shape [batch_size, sequence_length, embedding_size].
# 
# TensorFlow’s convolutional conv2d operation expects a 4-dimensional tensor with dimensions corresponding to batch, width, height and channel. The result of our embedding doesn’t contain the channel dimension, so we add it manually, leaving us with a layer of shape [None, sequence_length, embedding_size, 1].

# In[64]:


embedded_chars = embedding_layer(input_x)

embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)


# In[65]:


pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):
    # Convolution Layer
    filter_shape = [filter_size, embedding_size, 1, num_filters]
    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
    conv = tf.nn.conv2d(
        embedded_chars_expanded,
        W,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv")
    # Apply nonlinearity
    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
    # Max-pooling over the outputs
    pooled = tf.nn.max_pool(
        h,
        ksize=[1, sequence_length - filter_size + 1, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID',
        name="pool")
    pooled_outputs.append(pooled)


# In[66]:


print(pooled_outputs)


# In[67]:


# Combine all the pooled features
num_filters_total = num_filters * len(filter_sizes)
h_pool = tf.concat(pooled_outputs, len(filter_sizes))
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])


# ### Dropout
# Dropout is the perhaps most popular method to regularize convolutional neural networks. The idea behind dropout is simple. A dropout layer stochastically “disables” a fraction of its neurons. This prevent neurons from co-adapting and forces them to learn individually useful features. The fraction of neurons we keep enabled is defined by the dropout_keep_prob input to our network. We set this to something like 0.5 during training, and to 1 (disable dropout) during evaluation.

# In[68]:


# Add drop out
h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)


# ### Readout Layer

# In[69]:


W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

raw_scores = tf.matmul(h_drop, W) + b
normalized_scores = tf.nn.softmax(raw_scores)
predictions = tf.argmax(raw_scores, 1)


# ### Loss and Accuracy

# In[70]:


losses = tf.nn.softmax_cross_entropy_with_logits(logits=raw_scores, labels=input_y)
cross_entropy = tf.reduce_mean(losses)

correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


# ### Train and Evaluate the Model

# In[71]:


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


# In[72]:


train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


# In[73]:


probabilities = np.zeros((padded_encoded_test_text.shape[0], 3))
print("Training Started")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    batches = batch_iter(
            list(zip(x_train, y_train)), batch_size, num_epochs)
    
    for batch in batches:
        x_batch, y_batch = zip(*batch)
        #if i % 10 == 0:
            #train_accuracy = accuracy.eval(feed_dict={input_x: x_batch, input_y: y_batch, dropout_keep_prob: 1.0})
            #print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={input_x: x_batch, input_y: y_batch, dropout_keep_prob: 1.0})
    
    probabilities = sess.run(normalized_scores, feed_dict={input_x: padded_encoded_test_text, dropout_keep_prob: 1.0})


# In[74]:


test_id = test.as_matrix()[:,0]
out = np.rec.fromarrays([test_id, probabilities[:,0], probabilities[:,1], probabilities[:,2]])
print(out)


# In[75]:


# Produce csv
import csv
with open('glove_tf1C_10_64_23.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
    wr.writerow(('id', 'EAP','HPL','MWS'))
    wr.writerows(out)

