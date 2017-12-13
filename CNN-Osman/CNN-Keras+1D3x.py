
# coding: utf-8

# In[1]:


import pandas
import numpy as np
import matplotlib.pyplot as plt
import csv
np.set_printoptions(15, suppress=True)
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
#print(train)
#print(test_corpus)
#print(test)


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

#print(train_text[0])
#print(len(train_text_length))    
#print(test_text[0])
#print(len(test_text_length))


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
onehot_train_labels = to_categorical(train_targets)

#print(train_labels)
#print(onehot_train_labels)


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

#print(padded_encoded_train_text[0])
#print(padded_encoded_train_text.shape)
#print(padded_encoded_test_text[0])
#print(padded_encoded_test_text.shape)


# ## Embedding Layer

# In[10]:


# Load the whole embedding into memory
embeddings_index = dict()
f = open('resource/glovew2v.txt',encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))


# In[11]:


# Dimensionality of each embedding
embedding_size = len(embeddings_index.get('the'))


# In[12]:


vocabulary_size = len(t.word_index) + 1

# create a embedding matrix for words in training docs
embedding_matrix = np.zeros((vocabulary_size, embedding_size))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# In[13]:


from keras.layers import Embedding

print(vocabulary_size)

# Input shape to embedding layer: 2D tensor - (batch_size, sequence_length)
# Output shape of embedding layer: 3D tensor - (batch_size, sequence_length, output_dim)
embedding_layer = Embedding(vocabulary_size,
                           embedding_size,
                           weights=[embedding_matrix],
                           input_length=sequence_length,
                           trainable=False)


# In[14]:


x_train = padded_encoded_train_text
y_train = onehot_train_labels


# In[45]:


from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, Input
from keras.layers import Dense, Dropout, Activation, Flatten

num_classes = 3
batch_size = 64
num_filters = 250
filter_size = 3
num_epochs = 1

sequence_input = Input(shape=(sequence_length,), dtype='int32')
print(sequence_input.shape)

#sequence_input = padded_encoded_train_text
embedded_sequences = embedding_layer(sequence_input)
print(embedded_sequences.shape)


x = Conv1D(num_filters, filter_size, activation='relu')(embedded_sequences)
x = MaxPooling1D(filter_size)(x)

x = Conv1D(num_filters, filter_size, activation='relu')(x)
x = MaxPooling1D(filter_size)(x)

x = Conv1D(num_filters, filter_size, activation='relu')(x)
x = MaxPooling1D(filter_size)(x)  # global max pooling

x = Flatten()(x)
x = Dense(batch_size, activation='relu')(x)
preds = Dense(num_classes, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# happy learning!
model.fit(x_train, y_train, validation_split=0.0, shuffle=True,
          epochs=num_epochs, batch_size=batch_size)


# In[46]:


propabilities = model.predict(x=padded_encoded_test_text, batch_size=batch_size, verbose=0)


# In[47]:


print(propabilities.shape)


# In[48]:


test_id = test.as_matrix()[:,0]
#print(test_id)


# In[49]:


# Format csv output
#idx = np.arange(test_label.size, dtype=np.int16)
out = np.rec.fromarrays([test_id, propabilities[:,0], propabilities[:,1], propabilities[:,2]])
print(out)


# In[50]:


# Produce csv
with open('glove_keras3C_1_64_3.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
    wr.writerow(('id', 'EAP','HPL','MWS'))
    wr.writerows(out)

