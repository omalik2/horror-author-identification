import pickle
import csv
import numpy as np
import string
import operator


def predictClass(authors, text, vocab, prior, conditionalProb):
    score = []
    for c in authors:
      tempScore = prior[c]
      segment = text.translate(None, string.punctuation)
      for word in segment.split():
        if(word in vocab.keys()):
            wordIdx = vocab.keys().index(word)
            classIdx = authors.index(c)
            tempScore *= conditionalProb[wordIdx, classIdx]
      score.append(tempScore)
      
    authorIdx = score.index(max(score))
    return authors[authorIdx]

testPath = './Data/sampleTest.csv'       
        
#import probabiliie
probFileName = 'naiveBayesProbs'
probsFileObj = open(probFileName, 'r')
probs = pickle.load(probsFileObj)

author = probs['authors']
condProb = probs['conditional']
prior = probs['prior']
vocab = probs['vocab']

#newA = dict(sorted(vocab.iteritems(), key=operator.itemgetter(1), reverse=False)[:30])
#print newA



with open(testPath, 'rb') as csvfile:
    print author
    csvIter = csv.reader(csvfile)
    count = 0
    correct = 0
    for row in csvIter:
        pred = predictClass(author, row[1], vocab, prior, condProb)   
        count +=1
        if pred == row[2]:
            correct += 1
            
    print float(correct) * 100.0/float(count)        
       
        