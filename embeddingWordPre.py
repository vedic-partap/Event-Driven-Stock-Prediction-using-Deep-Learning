from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import os
import numpy as np
from nltk.corpus import reuters


import json
# Taking Out Word Vectors from standard Glove WV of stanford for our vocabulary
def get_reuters_data(n_vocab):
    # return variables
    sentences = []
    word2idx = {'START': 0, 'END': 1}
    idx2word = ['START', 'END']
    current_idx = 2
    word_idx_count = {0: float('inf'), 1: float('inf')}
    tag = 0
    for field in reuters.fileids():
        sentence = reuters.words(field)
        tokens = [unify_word(t) for t in sentence]
        for t in tokens:
            if t not in word2idx:
                word2idx[t] = current_idx
                idx2word.append(t)
                current_idx += 1
            idx = word2idx[t]
            word_idx_count[idx] = word_idx_count.get(idx, 0) + 1
        sentence_by_idx = [word2idx[t] for t in tokens]
        sentences.append(sentence_by_idx)
        tag += 1
        print(tag)

    # restrict vocab size
    sorted_word_idx_count = sorted(word_idx_count.items(), key=operator.itemgetter(1), reverse=True)
    word2idx_small = {}
    new_idx = 0
    idx_new_idx_map = {}
    for idx, count in sorted_word_idx_count[:n_vocab]:
        word = idx2word[idx]
        print( word, count)
        word2idx_small[word] = new_idx
        idx_new_idx_map[idx] = new_idx
        new_idx += 1
    # let 'unknown' be the last token
    word2idx_small['UNKNOWN'] = new_idx
    unknown = new_idx

    # map old idx to new idx
    sentences_small = []
    for sentence in sentences:
        if len(sentence) > 1:
            new_sentence = [idx_new_idx_map[idx] if idx in idx_new_idx_map else unknown for idx in sentence]
            sentences_small.append(new_sentence)

    return sentences_small, word2idx_small

def loadWordToIndex():
    w2i_file = './input/word2idx.json'
    sen = './input/sentences.json'
    if not os.path.isfile(w2i_file):
        sentences, word2idx = get_reuters_data(n_vocab=2000)
        with open(w2i_file, 'w') as f:
            json.dump(word2idx, f)
        with open(sen, 'w') as f:
            json.dump(sentences, f)
    else:
        with open(w2i_file) as data_file:
            word2idx = json.load(data_file)
        with open(sen) as data_file:
            sentences = json.load(data_file)
    return sentences,word2idx

def makeEmbedding(glove_file,tmp_file):
    #function to load vocabulary
    _,word2idx = loadWordToIndex()

    if not os.path.isfile(tmp_file):
        glove2word2vec(glove_file, tmp_file)

    model = KeyedVectors.load_word2vec_format(tmp_file)
    model = model.wv
    wordVecDim = model.vector_size

    wordEmbeddingsVec = np.zeros((len(word2idx),wordVecDim))

    for t in word2idx:
        index = word2idx[t]
        if t in model.vocab:
            wordEmbeddingsVec[index] = np.array(model[t])
        else:
            wordEmbeddingsVec[index] = np.array(model['unk'])

    print(wordEmbeddingsVec.shape)
    fileName = './input/wordEmbeddingsVocab'+ '.csv'
    with open(fileName, 'wb') as file:
        np.savetxt(file, wordEmbeddingsVec, fmt='%.5f')

if __name__ == "__main__":
    glove_file = "./input/glove.6B.100d.txt"
    #file where wordToVec model of glove will be stored
    tmp_file = "./input/stanfordglove_word2vec.txt"
    makeEmbedding(glove_file,tmp_file)
