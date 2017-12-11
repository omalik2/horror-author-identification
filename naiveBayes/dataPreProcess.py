# this will preprocess the data for learning the model
import numpy as np
import csv
import string
import pickle
import operator

def generateVocabDict(trainSet):
    vocab = {}
    trainData = {}
    with open(trainSet, 'rb') as csvfile:
        csvIter = csv.reader(csvfile)
        csvIter.next();
        for row in csvIter:
            author = row[2]

            
            segment = row[1];
            segment = segment.translate(None, string.punctuation)
            
            if author not in trainData:
                trainData[author] = [segment]
            else:
                trainData[author].append(segment)
            
            for word in segment.split():
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] += 1
    #vocab = dict(sorted(vocab.iteritems(), key=operator.itemgetter(1), reverse=True)[:5000])                
    return vocab, trainData
            

def calcPrior(trainSet):
    prior = {}
    numOfDocs = 0
    with open(trainSet, 'rb') as csvfile:
        csvIter = csv.reader(csvfile)
        csvIter.next();
        for row in csvIter:
            author = row[2]
            if author not in prior:
                prior[author] = 1
            else:
                prior[author] += 1
            
            numOfDocs += 1
        
    for key in prior:
        if prior[key] > 0:
            prior[key] = float(prior[key]) / float(numOfDocs)
    
    return prior
      
def trainNaiveBayes(authorSet, trainDataPath):
    # generate vocabulary
    vocab, trainSet = generateVocabDict(trainDataPath)

    #calculate prior for all authors
    prior = calcPrior(trainDataPath)

    #calculate conditional probability for all authors
    condProb = np.zeros((len(vocab),3))
    print len(vocab)
    authorIdx = 0
    for author in authorSet: # loop through authors
        print "Calculating Conditional Probabiliy for Author: " + author
        textList = trainSet[author]
        totalWordCount = 0
        #calculate total word count
        for text in textList:
            totalWordCount += len(text.split())
        
        idx = 0
        for word in vocab:
            sum = 0
            for text in textList:
                text = text.translate(None, string.punctuation)
                sum += text.split().count(word)
               
            
            #calculate conditional probablility
            condProb[idx,authorIdx] = (float(sum + 1) / float(totalWordCount + len(vocab)))  
            #print condProb[idx,authorIdx]
            idx += 1
        authorIdx += 1
    return condProb, prior, vocab
            
          

trainPath = './Data/train.csv'
testPath = './Data/test.csv'
outFileName = 'naiveBayesProbs'
authors = ['MWS','EAP','HPL']

condProb, prior, vocab = trainNaiveBayes(authors, trainPath)

probabilities = {'authors': authors, 'conditional' : condProb, 'prior' : prior, 'vocab' : vocab}

print "pickling data"

outFileObj = open(outFileName, 'wb')
pickle.dump(probabilities, outFileObj)
outFileObj.close()







