from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import matplotlib.pyplot as plt

number_of_output = 2001
samples_per_epoch = 0
nb_val_samples = 0

with open('train.txt', 'r') as file:
    vectorizer = CountVectorizer(max_features=2000,stop_words = 'english')
    corpus_train = file.readlines()
    tokenizer = RegexpTokenizer(r'\w+')
    for i in range(len(corpus_train)):
        corpus_train[i] = ' '.join(tokenizer.tokenize(corpus_train[i]))
    X =vectorizer.fit_transform(corpus_train)


with open('test.txt', 'r') as file:
    corpus_test = file.readlines()
    for i in range(0, len(corpus_test)):
        corpus_test[i] = ' '.join(tokenizer.tokenize(corpus_test[i]))
    X_test = vectorizer.fit_transform(corpus_test)



desired_words=vectorizer.get_feature_names()

for i in range(0, len(corpus_train)):
    corpus_train[i] = '<s> ' + corpus_train[i] + ' </s>'

for i in range(0, len(corpus_test)):
    corpus_test[i] = '<s> ' + corpus_test[i] + ' </s>'



from nltk import ngrams
fourgrams_train = set()
fourgrams_test = set()


for sentence in corpus_train:
    tmp = ngrams(sentence.split(), 5)
    for _ in tmp:
        samples_per_epoch += 1
    fourgrams_train.add(ngrams(sentence.split(), 5))



for sentence in corpus_test:
    tmp = ngrams(sentence.split(), 5)
    for _ in tmp:
        nb_val_samples += 1
    fourgrams_test.add(ngrams(sentence.split(), 5))



word2vec = {}
with open('glove.6B.50d.txt') as file:
    for line in file.readlines():
        tmp = line.split()
        word = tmp[0]
        score = tmp[1:]
        word2vec[word] = score



def get_word_score(word):
    lemmatizer = WordNetLemmatizer()
    if word == '<s>':
        return word2vec['<']
    elif word == '</s>':
        return word2vec['>']
    elif word in word2vec:
        return word2vec[word]
    elif lemmatizer.lemmatize(word) in word2vec:
        return word2vec[lemmatizer.lemmatize(word)]

    else:
        return word2vec['unknown']



def generate_one_train():
    while True:
        fourgrams =set()
        for sentence in corpus_train:
            fourgrams.add(ngrams(sentence.split(), 5))
        for g in fourgrams:
            for t in g:
                net_in = []

                for train in t[0:4]:
                    net_in.extend(get_word_score(train))

                if t[4] in desired_words:
                    index = desired_words.index(t[4])
                    target = np.zeros(shape=number_of_output,dtype = np.int8)
                    target[index] = 1
                else:
                    target = np.zeros(shape=number_of_output,dtype = np.int8)
                    target[2000] = 1
                yield np.array(net_in, dtype=np.float32).reshape(1,200), np.array(target).reshape(1,2001)


def generate_one_test():
    while True:
        fourgrams_test = set()
        for sentence in corpus_test:
            fourgrams_test.add(ngrams(sentence.split(), 5))
        for g in fourgrams_test:
            for t in g:
                net_in = []
                # first three word of tuple are our train
                for train in t[0:4]:
                    net_in.extend(get_word_score(train))


                if t[4] in desired_words:
                    index = desired_words.index(t[4])
                    target = np.zeros(shape=number_of_output,dtype = np.int8)
                    target[index] = 1
                else:
                    target = np.zeros(shape=number_of_output,dtype = np.int8)
                    target[2000] = 1
                return np.array(net_in, dtype=np.float32).reshape(1,200), np.array(target).reshape(1,2001)




import math

# corpus is a list. each item of the list is one line of corpus
def compute_perplexity(corpus):
    total_perplexity = 0
    for sentence in corpus:
        fourgram_sentence = ngrams(sentence.split(), 5)
        sentence_perplexity = None
        for t in fourgram_sentence:
            prob = 1
            if len(t) < 5:
                continue
            net_in = []
            # first three word of tuple are our train
            for train in t[0:4]:
                net_in.extend(get_word_score(train))
            if t[4] in desired_words:
                index_of_target_word = desired_words.index(t[4])
            else:
                index_of_target_word = 2000
            net_out = classifier.predict_proba(np.array(net_in, dtype=np.float32).reshape(1,200))
            prob_target_word = net_out[0][index_of_target_word]
            prob = prob*prob_target_word
        sentence_perplexity = math.sqrt(1/prob)
        total_perplexity += sentence_perplexity
    return total_perplexity/len(corpus)

#from tensorflow.keras import backend as k
#def perplexity(y_true,y_pred):
#    cross_entropy=k.categorical_crossentropy(y_true,y_pred)
#    return k.pow(2.0,cross_entropy)



from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

opt = SGD(lr=0.02)

classifier = Sequential()
classifier.add(Dense(units = 35,input_dim=200, activation='relu', kernel_initializer='uniform'))

classifier.add(Dense(units=2001, activation='softmax'))

classifier.compile( loss='categorical_crossentropy', metrics=['accuracy'],optimizer=opt)
#classifier.compile( loss='categorical_crossentropy', metrics=['accuracy',perplexity],optimizer=opt)

history=classifier.fit_generator(
        generate_one_train(),
        validation_data=generate_one_test(),
        samples_per_epoch=samples_per_epoch,
        nb_val_samples=nb_val_samples,
        nb_epoch=1)


import pickle
with open('classifier', 'wb') as f:
    pickle.dump(classifier, f)



# plt.plot(history.history['accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# # plt.legend(['train', 'test'], loc='upper left')
# plt.show()




print('train:', compute_perplexity(corpus_train))
print('test:', compute_perplexity(corpus_test))
