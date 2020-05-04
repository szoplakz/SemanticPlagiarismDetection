import main.Fileparser as Fileparser
import re
import numpy as np
import math
Fingerprint_size=16384
Fingerprint_width=128

def cleanFingerPrints(fingerprints):
    # removes unnecessary information from the fingerprint files that isn't directly related to the fingerprints themselves
    fingerPrintList=[]
    fingerprints[0]=re.sub("[^0-9]","",fingerprints[0])
    fingerprints[len(fingerprints)-1]=re.sub("[^0-9]","",fingerprints[len(fingerprints)-1])
    fPrint=[]
    for p in fingerprints:
        p=re.sub(r"\s+", "", p)
        fPrint.append(p)
    return fPrint


def splitFingerprintIntoParts(indices,numParts):
    # splits a fingerprint into multiple smaller arrays
    partLength=math.ceil(16384/numParts)
    indices.sort()
    offset=0
    printList=[[],[],[],[]]
    j=0
    for i in range(numParts):
        printPart=[]
        offset+=partLength
        while (j<len(indices) and indices[j]<offset):
            index=int(indices[j]-(offset-partLength))
            printPart.append(index)
            j+=1
        printList[i]=printPart
    return printList, numParts, partLength

def mergeSplitIndices(activeIndices,predictiveIndices,numParts):
    # merges multiple smaller arrays that have been split into a single fingeprint
    aI =  []
    pI=[]
    partLength = math.ceil(16384 / numParts)
    for i in range(len(activeIndices)/numParts):
        active=[]
        predictive=[]
        for j in range(numParts):
            indices=[(numParts*partLength)+int(index) for index in activeIndices[i*numParts+j]]
            active.extend(indices)
            indices=[(numParts*partLength)+int(index) for index in predictiveIndices[i*numParts+j]]
            predictive.extend(indices)
        aI.append(active)
        pI.append(predictive)
    return aI, pI

def getFingerprintOverlap(print1,print2):
        # returns the indices that have a value of 1 from both fingerpritns
        temp = set(print2)
        overlap = [value for value in print1 if value in temp]
        return overlap

def SDRFromFingerprint(fPrint):
    # creates an SDR vector from fingerprints indices
    SDR = np.zeros((Fingerprint_width, Fingerprint_width),dtype=np.int8)
    for p in fPrint:
        idx=int(p)
        rowIdx=idx//Fingerprint_width
        columnIdx= idx % Fingerprint_width
        SDR[rowIdx][columnIdx]=1;
    return SDR


def getSimilarityMetrics(print1,print2):
    # returns the euclidean distance, cosine similarity, jaccard index and jaccard distance from the comparison of two fingerprints
    overlapSize = len(getFingerprintOverlap(print1,print2))
    length1 = len(list(print1))
    length2 = len(list(print2))
    if(length1==0 or length2==0):
        return 0,0,0
    unionSize = length1+length2-overlapSize
    dissimilaritySize = unionSize-overlapSize
    euclideanDistance= math.sqrt(dissimilaritySize)

    cosineSimilarity = overlapSize/(math.sqrt(length1)*math.sqrt(length2))
    jaccardIndex=overlapSize/unionSize
    return euclideanDistance, cosineSimilarity, jaccardIndex

def aggregateFingerPrints(printList):
    # returns a dictionary of non-zero indices and their number of occurences within a list of fingerprints
    indexDict = {}
    for fPrint in printList:
        for p in fPrint:
             if p in indexDict:
                 indexDict[p]=(int(indexDict[p])+1)
             else:
                 indexDict.update({p:1})
    return indexDict

def kSparsify(indexDict, k):
    # sparsifies the aggregated fingerprint representation to create a merged print
    fPrint=[]
    for idx in indexDict:
        if(int(indexDict[idx])>=k):
            fPrint.append(idx)
    fPrint.sort(key=int)
    return fPrint

def mergePrints(printList,k):
    # merges multiple fingerprints into a single print by aggregation and sparsification
    indexDict = aggregateFingerPrints(printList)
    return kSparsify(indexDict, k)

def saveCleanFingerPrints(fingerprints,path):
    # saves the clean fingerprints to a file
    with open(path, 'w') as cleanFile:
        for i in range(len(fingerprints)):
            prints = cleanFingerPrints(fingerprints[i])
            for j in range(len(fingerprints[i])):
                cleanFile.write((prints[j]))
                cleanFile.write(",")
            cleanFile.write("|")

def generateWindowFingerprints(fingerprints,windowSize):
    # for a list of fingerprints, generates merged prints from a window of fingerprints
    windowPrints = []
    windowScores = []
    for i in range(windowSize):
        windowScores.append(fingerprints[i])
        windowPrints.append(mergePrints(windowScores,math.ceil(math.sqrt(len(windowScores)))))

    for i in range(windowSize, len(fingerprints)):
        windowScores.pop(0)
        windowScores.append(fingerprints[i])
        windowPrints.append(mergePrints(windowScores, math.ceil(math.sqrt(windowSize))))
    return windowPrints

def generateWindowOverlapFingerprints(fingerprints,windowSize):
    # for a list of fingerprints, generates merged prints from the window fingerprints that came before it and the fingerprint itself
    overlapPrints=[]
    windowPrints = generateWindowFingerprints(fingerprints,windowSize)
    overlapPrints.append(windowPrints[0])
    for i in range(1,len(fingerprints)):
        printsTomerge=[fingerprints[i],windowPrints[i-1]]
        overlapPrints.append(mergePrints(printsTomerge,2))
    return overlapPrints

def calculateWindowOverlapFingerprints(fingerprints,windowSize):
    # calculates the similarity measures gained by comparing the window fingerprints that came before it and the fingerprint itself
    similarityMeasuresList=[0,0,0]
    windowPrints = generateWindowFingerprints(fingerprints,windowSize)
    for i in range(1,len(fingerprints)):
        euclideanDistance, cosineSimilarity, jaccardIndex=\
            getSimilarityMetrics(fingerprints[i],windowPrints[i-1])
        similarityMeasuresList.append(euclideanDistance)
        similarityMeasuresList.append(cosineSimilarity)
        similarityMeasuresList.append(jaccardIndex)
    return similarityMeasuresList

def getFingerprintOfDoc(fingerprints):
    # creates a fingerprint of the entire document
    fingerprint_list=mergePrints(fingerprints,math.ceil(math.sqrt(len(fingerprints))))
    return fingerprint_list

def getFingerprintsOfSection(fingerprints,startId,endId):
    # creates a fingerprint of a portential part
    printList=fingerprints[startId:endId]
    fingerprint_list=mergePrints(printList,math.ceil(math.sqrt(len(printList))))
    return fingerprint_list

def calculateOverlapScoresOfParts(fingerprints,indices):
    # uses the similarity metrics to compare the print individual parts with the print of the document
    startIndices=[]
    endIndices=[]
    euclideanList=[]
    cosineList=[]
    jaccardList=[]
    startIndices.append(0)
    endIndices.append(indices[0])
    for i in range(1,len(indices)):
        startIndices.append(indices[i-1])
        endIndices.append(indices[i])
    startIndices.append(indices[len(indices)-1])
    endIndices.append(len(fingerprints))
    docPrint=getFingerprintOfDoc(fingerprints)
    for i in range(len(endIndices)):
        sectionPrint=getFingerprintsOfSection(fingerprints,startIndices[i],endIndices[i])
        euclidean,cosine,jaccard=getSimilarityMetrics(sectionPrint,docPrint)
        euclideanList.append(euclidean)
        cosineList.append(cosine)
        jaccardList.append(jaccard)
    return euclideanList,cosineList,jaccardList

def writeSimilarityMeasures(idx,windowsize):
    # writes the similarity metrics of the classifier to a file
    f_path = "C:\DiplomaProject\CleanFingerprints\Fingerprints" + str(idx) + ".txt"
    fingerprints=Fileparser.get_clean_fingerprints_from_file(f_path)
    overlapFingerprints = calculateWindowOverlapFingerprints(fingerprints,windowsize)
    goal_path = "C:\DiplomaProject\SimilarityMetrics\SimilarityMetrics" + str(idx) + ".txt"
    with open(goal_path, 'w', encoding="utf-8", errors='ignore') as f:
        f.truncate(0)
        for j in range(int(len(overlapFingerprints)/3)):
            f.write(str(overlapFingerprints[j*3])+",")
            f.write(str(overlapFingerprints[j*3+1])+",")
            f.write(str(overlapFingerprints[j*3+2]))
            f.write("|")
        f.close()

def readSimilarities(idx):
    # reads the similarity metrics of the classifier to a file
    euclidean=[]
    cosine=[]
    jaccard=[]
    path = "C:\DiplomaProject\SimilarityMetrics\SimilarityMetrics" + str(idx) + ".txt"
    with open(path, encoding="utf-8", errors='ignore') as f:
        text = f.read()
        indices = text.split("|")
        indices.pop()
        for i in indices:
            measures = i.split(",")
            euclidean.append(float(measures[0]))
            cosine.append(float(measures[1]))
            jaccard.append(float(measures[2]))
    return euclidean,cosine,jaccard

for i in range(39,40):
    writeSimilarityMeasures(i,5)