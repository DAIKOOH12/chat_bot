
# coding: utf-8

# In[1]:


import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
from pyvi import ViTokenizer
from pyvi import ViPosTagger
from pyvi import ViUtils

stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow as tf
import random


# In[2]:


import json
with open('data/intents.json') as json_data:
    intents = json.load(json_data)


# In[42]:


def ngrams(str, n):
    tokens = str.split(' ')
    arr = []
    for i in range(len(tokens)):
        new_str = ''
        if i == 0 and n>1:
            new_str = '_'
            for j in range(n):
                if j < n - 1:
                    if (i + j) <= len(tokens):
                        new_str += ' '+tokens[i+j]
                    else:
                        new_str += ' _'
        else:
            for j in range(n):
                if j < n:
                    if (i + j) < len(tokens):
                        if j == 0:
                            new_str += tokens[i+j]
                        else:
                            new_str += ' '+tokens[i+j]
                    else:
                        new_str += ' _'
        arr.append(new_str)
    return arr


# In[43]:


ngrams('a b c d e f g h', 4)


# In[3]:


words = []
classes = []
documents = []
ignore_words = ['?', 'và', 'à', 'ừ', 'ạ', 'vì', 'từng', 'một_cách', 'nào','như_nào', 'sao' ,'làm_sao', 'thế_nào', 'có', 'những', 'của']

for intent in intents['intents']: #danh sách các mô hình câu hỏi tương ứng của intents['intents']
    for pattern in intent['patterns']: #danh sách các câu hỏi (patterns) của từng intent

        # w = nltk.word_tokenize(pattern) #tách pattern thành các từ riêng biệt
        # w = ViPosTagger.postagging(pattern.lower())
        w = ViTokenizer.tokenize(pattern.lower()).split()
        words.extend(w)

        documents.append((w, intent['tag'])) #thêm cặp (câu hỏi, lớp) vào documents
        if intent['tag'] not in classes: #nếu lớp (tag) của intent đang xét chưa có trong classes thì thêm vào classes
            classes.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
# words = [ViPosTagger.postagging(w.lower()) for w in words if w not in ignore_words][0]


words = sorted(list(set(words)))



classes = sorted(list(set(classes)))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)


# In[4]:


#Create training data
training = []
output = []

output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    # patten_words = [ViPosTagger.postagging(word.lower()) for word in pattern_words][0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
# training = np.array(training)
# #
# train_x = list(training[:,0])
# train_y = list(training[:,1])

train_x =[] #Lưu tất cả các vecto dac trung
train_y =[] #Lưu các vecto dau ra
for item in training:
    train_x.append(item[0])
    train_y.append(item[1])

# train_x = np.array(train_x)
# train_y = np.array(train_y)

print(train_x)
print(train_y)


# In[5]:


print(train_x[1])
print(train_y[1])


# In[6]:


tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('models/model.tflearn')


# In[8]:


import pickle
pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "models/training_data", "wb" ) )

