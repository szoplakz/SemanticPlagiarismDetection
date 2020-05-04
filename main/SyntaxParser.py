import main.Fileparser as Fileparser
import numpy as np
import operator
import math


def merge_sentences(sentenceList):
    # merges multiple sentences from a list into a singular sentence
    finalString=""
    for sent in sentenceList:
        finalString+=str(sent)
        finalString+=" "
    return finalString

def getWordFrequencies(text):
    # creates a dictionary of the frequencies of all texts
    text = " ".join(text.split())
    text=Fileparser.remove_stopwords(text)
    words=text.split()
    freqDict={}
    for w in words:
        if w in freqDict:
            freqDict[w] = (int(freqDict[w]) + 1)
        else:
            freqDict.update({w: 1})
    return freqDict



def calculateRelationalFrequencyOfSentence(corpus):
    # calculates the mean, lowest and highest relational frequencies of the sentence
    fullText = merge_sentences(corpus)
    corpusFrequencies=getWordFrequencies(fullText)
    biggestFrequency=max(corpusFrequencies.items(), key=operator.itemgetter(1))[1]
    meanFreqs=[]
    lowestFreqs=[]
    upperFreqs=[]
    for i in range(len(corpus)):
        sentFrequencies=getWordFrequencies(corpus[i])
        relationalFrequencies=[]
        freqSum=0
        for w in sentFrequencies:
            freqVal = math.log(biggestFrequency/(corpusFrequencies[w]-sentFrequencies[w]+1),2)
            relationalFrequencies.append(freqVal)
            freqSum+=freqVal

        if (len(relationalFrequencies)==0):
            meanFreqs.append(0)
            lowestFreqs.append(0)
            upperFreqs.append(0)
        else:
            meanFreqs.append(freqSum/len(relationalFrequencies))
            lowestFreqs.append(min(relationalFrequencies))
            upperFreqs.append(max(relationalFrequencies))
    return meanFreqs, lowestFreqs, upperFreqs

def calculateRelationalFrequenciesOfPart(startId,partLength,corpus):
    # calculates the mean, lowest and highest relational frequencies of a part of text
    fullText = merge_sentences(corpus)
    corpusFrequencies = getWordFrequencies(fullText)
    biggestFrequency = max(corpusFrequencies.items(), key=operator.itemgetter(1))[1]

    sentences=[]
    for i in range(partLength):
        sentences.append(corpus[startId+i])
    mergedSents=merge_sentences(sentences)
    sentFrequencies = getWordFrequencies(mergedSents)
    relationalFrequencies = []
    freqSum = 0
    for w in sentFrequencies:
        freqVal = math.log(biggestFrequency / (corpusFrequencies[w] - sentFrequencies[w] + 1), 2)
        relationalFrequencies.append(freqVal)
        freqSum += freqVal

    if (len(relationalFrequencies) == 0):
        meanFreq=0
        lowestFreq=0
        upperFreq=0
    else:
        meanFreq=(freqSum / len(relationalFrequencies))
        lowestFreq=min(relationalFrequencies)
        upperFreq=max(relationalFrequencies)
    return meanFreq,lowestFreq,upperFreq




def relationalFrequenciesToFile(idx):
    # calculates the relational frequenies of sentences and writes them to a file
    corpusPath = "C:\DiplomaProject\OriginalCorpus\Corpus" + str(idx) + ".txt"
    corpus = Fileparser.get_corpus_from_file(corpusPath)
    sents=[corpus[0],corpus[1],corpus[2]]
    sentenceFives=[merge_sentences(sents)]
    sents.append(corpus[3])
    sentenceFives.append(merge_sentences(sents))
    for i in range(2,len(corpus)-2):
        sents=[corpus[i-2],corpus[i-1],corpus[i],corpus[i+1],corpus[i+2]]
        merged = merge_sentences(sents)
        sentenceFives.append(merged)
    finalIdx=len(corpus)-1
    sents=[corpus[finalIdx-3],corpus[finalIdx-2],corpus[finalIdx-1],corpus[finalIdx]]
    sentenceFives.append(merge_sentences(sents))
    sents.pop(0)
    sentenceFives.append(merge_sentences(sents))
    meanFreqs, lowestFreqs, upperFreqs = calculateRelationalFrequencyOfSentence(corpus)
    goal_path="C:\DiplomaProject\RelationalFrequncies\Frequncies" + str(idx) + ".txt"
    with open(goal_path, 'w', encoding="utf-8", errors='ignore') as f:
        f.truncate(0)
        for j in range(len(meanFreqs)):
            f.write(str(meanFreqs[j])+",")
            f.write(str(lowestFreqs[j])+",")
            f.write(str(upperFreqs[j]))
            f.write("|")
        f.close()

def readRelationalFreqs(idx):
    # reads the relational frequencies of a document from a file
    path = "C:\DiplomaProject\RelationalFrequncies\Frequncies" + str(idx) + ".txt"
    meanFreqs=[]
    lowestFreqs=[]
    upperFreqs=[]
    with open(path, encoding="utf-8", errors='ignore') as f:
        text = f.read()
        indices = text.split("|")
        indices.pop()
        for i in indices:
            freqs = i.split(",")
            meanFreqs.append(float(freqs[0]))
            lowestFreqs.append(float(freqs[1]))
            upperFreqs.append(float(freqs[2]))
    return meanFreqs,lowestFreqs,upperFreqs

def calculateAuthorStyle(idList,corpus):
    # calculates the author style  and individual scores of parts

    fullText = merge_sentences(corpus)
    corpusFrequencies = getWordFrequencies(fullText)
    authorScores=[]
    for i in range(1,len(idList)):
        sentList=[]
        for j in range(idList[i-1],idList[i]):
            sentList.append(corpus[j])
        mergedSents=merge_sentences(sentList)
        segmentFrequencies=getWordFrequencies(mergedSents)
        score=0
        for w in segmentFrequencies:
            score +=(corpusFrequencies[w]-segmentFrequencies[w])/(corpusFrequencies[w]+segmentFrequencies[w])
        authorScores.append(score)
    style=sum(authorScores)/(len(idList)-1)
    return authorScores,style


def authorStyleOfSentences(corpus):
    # calculates the author style difference of individual sentences
    idList = []
    for i in range(len(corpus)):
        idList.append(i)
    scores,style=calculateAuthorStyle(idList,corpus)
    differenceVals=[]
    for i in range(len(scores)):
        differenceVals.append(abs(scores[i]-style))
    return differenceVals

def authorStyleOfParts(startIds,corpus):
    # calculates the author style difference of parts - deprecated
    idList=[]
    for id in startIds:
        idList.append(id)
    idList.append(len(corpus)-1)
    scores, style = calculateAuthorStyle(idList, corpus)
    differenceVals = []
    for i in range(len(scores)):
        differenceVals.append(abs(scores[i] - style))
    return differenceVals


