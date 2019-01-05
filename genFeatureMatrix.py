#!/usr/bin/python
import json
import os
# import en
import datetime
import nltk
import numpy as np

def dateGenerator(numdays): # generate N days until now, eg [20151231, 20151230]
    base = datetime.datetime.today()
    date_list = [base - datetime.timedelta(days=x) for x in range(0, numdays)]
    for i in range(len(date_list)): date_list[i] = date_list[i].strftime("%Y%m%d")
    return set(date_list)

def unify_word(word): # went -> go, apples -> apple, BIG -> big
    # try: word = en.verb.present(word) # unify tense
    # except: pass
    # try: word = en.noun.singular(word) # unify noun
    # except: pass
    return word.lower()

def readGlove(we_file):
    wordVec = np.zeros([0,100])
    with open(we_file) as file:
        for line in file:
            line = line.strip().split()
            line = list(map(float,line))
            wordVec = np.vstack((wordVec,np.array(line).flatten()))
    return wordVec

def padding(sentencesVec, keepNum):
    shape = sentencesVec.shape[0]
    ownLen = sentencesVec.shape[1]
    if ownLen < keepNum:
        return np.hstack((np.zeros([shape, keepNum-ownLen]), sentencesVec)).flatten()
    else:
        return sentencesVec[:, -keepNum:].flatten()

def gen_FeatureMatrix(wordEmbedding, word2idx, priceDt, max_words=60, mtype="test",flag=0):
    # step 2, build feature matrix for training data
    loc = './input/'
    input_files = [f for f in os.listdir(loc) if f.startswith('news_reuters.csv')]
    #print(input_files)
    current_idx = 2
    dp = {} # only consider one news for a company everyday
    cnt = 0
    #testDates = dateGenerator(100)
    shape = wordEmbedding.shape[1]
    print(shape)
    features = np.zeros([0, max_words * shape])
    #print(features.shape)
    labels = []
    for file in input_files:
        count = 0 # Not more than 50k news
        with open(loc+file) as f:
            if mtype == 'test' and not flag:
                f.seek(125000,0)           # seek to end of file; f.seek(0, 2) is legal
            if mtype == 'validation' and not flag:
                f.seek(100000,0)
            if mtype == 'train' and flag :
                f.seek(50000,0)
            if mtype == 'test' and flag :
                f.seek(137500,0)
            if mtype == 'validation' and flag :
                f.seek(112500,0)
            for line in f:
                if mtype == 'test' and count == 12500: break
                if mtype == 'train' and count== 50000: break
                if mtype == 'validation' and count == 12500:break
                line = line.strip().split(',')
                if len(line) != 6: continue
                ticker, name, day, headline, body ,newsType= line
                #print("Toshal")
                if ticker not in priceDt: continue # skip if no corresponding company found
                if day not in priceDt[ticker]: continue # skip if no corresponding date found
                cnt += 1
                print(cnt)
                #if mtype == "test" and day not in testDates: continue
                #if mtype == "train" and day in testDates: continue
                # 2.1 tokenize sentense, check if the word belongs to the top words, unify the format of words
                tokens = nltk.word_tokenize(headline) + nltk.word_tokenize(body)
                tokens = [unify_word(t) for t in tokens]
                #tokens = [t for t in tokens if t in stopWords]
                #tokens = [t for t in tokens if t in topWords]
                # 2.2 create word2idx/idx2word list, and a list to count the occurence of words
                sentencesVec = np.zeros([shape, 0])
                for t in tokens:
                    if t not in word2idx: continue
                    sentencesVec = np.hstack((sentencesVec, np.matrix(wordEmbedding[word2idx[t]]).T))
                features = np.vstack((features, padding(sentencesVec, max_words)))
                count+=1 # increment news count
                labels.append(round(priceDt[ticker][day], 6))
    features = np.array(features)
    labels = np.matrix(labels)
    featureMatrix = np.concatenate((features, labels.T), axis=1)
    fileName = './input/featureMatrix_'+ str(flag) + "_" + mtype + '.csv'
    np.savetxt(fileName, featureMatrix, fmt="%s")

def build(wordEmbedding, w2i_file, max_words=60):
    with open('./input/stockPrices.json') as data_file:
        priceDt = json.load(data_file)
    with open(w2i_file) as data_file:
        word2idx = json.load(data_file)

    #gen_FeatureMatrix(wordEmbedding, word2idx, priceDt, max_words, "train",0)
    gen_FeatureMatrix(wordEmbedding, word2idx, priceDt, max_words, "validation",0)
    gen_FeatureMatrix(wordEmbedding, word2idx, priceDt, max_words, "test",0)
    # Making Additional Features if required
    #gen_FeatureMatrix(wordEmbedding, word2idx, priceDt, max_words, "train",1)
    gen_FeatureMatrix(wordEmbedding, word2idx, priceDt, max_words, "validation",1)
    gen_FeatureMatrix(wordEmbedding, word2idx, priceDt, max_words, "test",1)

def main(we, w2i_file):
    wordEmbedding = readGlove(we)
    build(wordEmbedding, w2i_file, 30)


if __name__ == "__main__":
    we = './input/wordEmbeddingsVocab.csv'
    w2i_file = "./input/word2idx.json"
    main(we, w2i_file)
