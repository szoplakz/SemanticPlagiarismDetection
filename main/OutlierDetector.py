import main.Fileparser as Fileparser
import main.RetinaOperations as RetinaOperations
import math
from sklearn import preprocessing
import numpy as np
from operator import itemgetter, attrgetter
from main.Fileparser import plagiarism
import main.SyntaxParser as SyntaxParser

class OffsetMetrics:
    # defines a class for the parameters of the classifier
    def __init__(self, id,endIdx,anomalyScore,likelyhoodScore,cosineDoc2Vec,euclideanDistance, cosineSimilarity, jaccardIndex,meanFreq,lowestFreq,upperFreq,authorStyleVals):
        self.id = id
        self.endIdx = endIdx
        self.anomalyScore = anomalyScore
        self.likelyhoodScore = likelyhoodScore
        self.cosineDoc2Vec=cosineDoc2Vec
        self.euclideanDistance=euclideanDistance
        self.cosineSimilarity=cosineSimilarity
        self.jaccardIndex=jaccardIndex
        self.meanFreq=meanFreq
        self.lowestFreq = lowestFreq
        self.upperFreq = upperFreq
        self.authorStyleVals=authorStyleVals

    def toString(self):
        print("----------------")
        print("sentenceNum: " + str(self.id))
        print("endIdx " + str(self.endIdx))
        #print("text " + str(self.text))
        #print("anomalyScore " + str(self.anomalyScore))
        print("likelyhoodScore " + str(self.likelyhoodScore))
        #print("cosineDoc2Vec " + str(self.cosineDoc2Vec))
        print("euclideanDistance " + str(self.euclideanDistance))
        print("cosineSimilarity " + str(self.cosineSimilarity))
        print("jaccardIndex " + str(self.jaccardIndex))
        print("meanFreq " + str(self.meanFreq))
        print("lowestFreq " + str(self.lowestFreq))
        print("upperFreq " + str(self.upperFreq))
        print("authorStyleDifferences " + str(self.authorStyleVals))

class PartMetric:
    # defines a class for the parameters of the regressor
    def __init__(self, startId,endId,startIndex,endIndex, partLength, cosineDoc2Vec,euclideanDistance, cosineSimilarity,
                 jaccardIndex,meanFreq,lowestFreq,upperFreq,meanStyle,partEuclid,partCosine,partJaccard,partMeanFreq,partLowestFreq,partUpperFreq ):
        self.startId = startId
        self.endId = endId
        self.startIndex = startIndex
        self.endIndex = endIndex
        self.partLength=partLength
        self.cosineDoc2Vec=cosineDoc2Vec
        self.euclideanDistance=euclideanDistance
        self.cosineSimilarity=cosineSimilarity
        self.jaccardIndex=jaccardIndex
        self.meanFreq=meanFreq
        self.lowestFreq = lowestFreq
        self.upperFreq = upperFreq
        self.meanStyle=meanStyle

        self.partEuclid=partEuclid
        self.partCosine = partCosine
        self.partJaccard = partJaccard
        self.partMeanFreq = partMeanFreq
        self.partLowestFreq = partLowestFreq
        self.partUpperFreq = partUpperFreq

class partStats:
    # a class for the potential parts
    def __init__(self, startId, endId, partLength, resultVal):
        self.startId = startId
        self.endId = endId
        self.resultVal = resultVal
        self.partLength=partLength

def getPartCorpusMetrics(idx,startIds,endIds):
    # collects and creates the input dataset for the regressor
    path = "C:\DiplomaProject\AlmarimiDocuments"
    documentList = Fileparser.extract_plagiarisms_from_files(path)
    text = Fileparser.extract_text_from_document(path, documentList[idx])
    start, end, sentences = Fileparser.split_into_sentences(text)
    originalCorpusPath = "C:\DiplomaProject\OriginalCorpus\Corpus" + str(idx) + ".txt"
    alamrimiCorpusPath = "C:\DiplomaProject\AlmarimiCorpus\Corpus" + str(idx) + ".txt"
    cosPath = "C:\DiplomaProject\CosineSimilarities\Cosine" + str(idx) + ".txt"

    f_path = "C:\DiplomaProject\CleanFingerprints\Fingerprints" + str(idx) + ".txt"
    index_path = "C:\DiplomaProject\AlmarimiIndices\Indices" + str(idx) + ".txt"
    SyntaxIds_path = "C:\DiplomaProject\SyntaxIds\Ids" + str(idx) + ".txt"
    SemanticIds_path = "C:\DiplomaProject\SemanticIds\Ids" + str(idx) + ".txt"
    SyntaxIds = Fileparser.get_indices_from_file(SyntaxIds_path)
    SemanticIds = Fileparser.get_indices_from_file(SemanticIds_path)
    originalCorpus = Fileparser.get_corpus_from_file(originalCorpusPath)
    cosineDoc2Vec = list(map(float, Fileparser.get_indices_from_file(cosPath)))
    fingerprints = Fileparser.get_clean_fingerprints_from_file(f_path)
    indices = list(map(float, Fileparser.get_indices_from_file(index_path)))
    euclidean, cosine, jaccard = RetinaOperations.readSimilarities(idx)
    euclidean = list(map(float, euclidean))
    cosine = list(map(float, cosine))
    jaccard = list(map(float, jaccard))
    meanFreq, lowestFreq, upperFreq = SyntaxParser.calculateRelationalFrequencyOfSentence(originalCorpus)
    meanFreq = list(map(float, meanFreq))
    lowestFreq = list(map(float, lowestFreq))
    upperFreq = list(map(float, upperFreq))
    authorStyleVals = SyntaxParser.authorStyleOfSentences(originalCorpus)
    cosineScores = []
    mean = []
    lowest = []
    upper = []
    authorStyle = []
    for i in range(len(SyntaxIds)):
        cosineScores.append(cosineDoc2Vec[int(SyntaxIds[i]) - 1])
        mean.append(meanFreq[int(SyntaxIds[i]) - 1])
        lowest.append(lowestFreq[int(SyntaxIds[i]) - 1])
        upper.append(upperFreq[int(SyntaxIds[i]) - 1])
        authorStyle.append(authorStyleVals[int(SyntaxIds[i]) - 1])

    euclid = []
    cos = []
    jacc = []

    for i in range(len(SemanticIds)):
        euclid.append(euclidean[int(SemanticIds[i]) - 1])
        cos.append(cosine[int(SemanticIds[i]) - 1])
        jacc.append(jaccard[int(SemanticIds[i]) - 1])

    partLengths=[]

    startIndices=[]
    endIndices=[]
    partMetrics = []
    if(len(startIds)==0):
        return partMetrics
    if(startIds[0]==0):

        partLengths.append(endIds[0]+1)
        startIndices.append(0)
        endIndices.append(end[endIds[0]])
    else:
        partLengths.append(endIds[0] - startIds[0] + 1)
        startIndices.append(end[startIds[0] - 1] + 1)
        endIndices.append(end[endIds[0]])

    for i in range(1,len(startIds)):
        partLengths.append(endIds[i]-startIds[i]+1)
        startIndices.append(end[startIds[i]-1]+1)
        endIndices.append(end[endIds[i]])
    print(len(partLengths))

    cosineDoc2Vec = []
    meanFreq = []
    lowestFreq = []
    upperFreq = []
    meanStyle = []
    euclidean = []
    cosine = []
    jaccard = []
    for i in range(len(partLengths)):
        cosineDoc2Vec.append(getMeanOfValues(startIds[i],partLengths[i],cosineScores))
        meanFreq.append(getMeanOfValues(startIds[i],partLengths[i],mean))
        lowestFreq.append(getMeanOfValues(startIds[i],partLengths[i],lowest))
        upperFreq.append(getMeanOfValues(startIds[i],partLengths[i],upper))
        euclidean.append(getMeanOfValues(startIds[i],partLengths[i],euclid))
        cosine.append(getMeanOfValues(startIds[i], partLengths[i], cos))
        jaccard.append(getMeanOfValues(startIds[i], partLengths[i], jacc))
        meanStyle.append(getMeanOfValues(startIds[i], partLengths[i], authorStyle))

    partEuclids,partCosines,partJaccards=RetinaOperations.calculateOverlapScoresOfParts(fingerprints,endIds)

    meanPartFreqs=[]
    lowestPartFreqs=[]
    upperPartFreqs=[]
    for i in range(len(partLengths)):
        meanPartFreq,lowestPartFreq,upperPartFreq=SyntaxParser.calculateRelationalFrequenciesOfPart(startIds[i],partLengths[i],sentences)
        meanPartFreqs.append(meanPartFreq)
        lowestPartFreqs.append(lowestPartFreq)
        upperPartFreqs.append(upperPartFreq)
    partMetrics=[]
    for i in range(len(partLengths)):
        metric=PartMetric(startIds[i],endIds[i],startIndices[i],endIndices[i], partLengths[i], cosineDoc2Vec[i],euclidean[i], cosine[i],
                 jaccard[i],meanFreq[i],lowestFreq[i],upperFreq[i],meanStyle[i],partEuclids[i],partCosines[i],partJaccards[i],
                meanPartFreqs[i],lowestPartFreqs[i],upperPartFreqs[i])
        partMetrics.append(metric)
    return partMetrics

def getMeanOfValues(startId,partLength,valuesToMerge):
    # a method to create a metric of a part by averaging the value across all of its sentences
    valueSum=0
    for i in range(partLength):
        id=startId+i
        valueSum+=valuesToMerge[id]
    return valueSum/partLength

def getOffsetCorpusMetrics(idx):
    # a method to get the metrics of the regressor
    path = "C:\DiplomaProject\AlmarimiDocuments"
    documentList = Fileparser.extract_plagiarisms_from_files(path)
    text = Fileparser.extract_text_from_document(path, documentList[idx])
    start, end, sentences = Fileparser.split_into_sentences(text)
    originalCorpusPath = "C:\DiplomaProject\OriginalCorpus\Corpus" + str(idx) + ".txt"
    alamrimiCorpusPath="C:\DiplomaProject\AlmarimiCorpus\Corpus" + str(idx) + ".txt"
    cosPath = "C:\DiplomaProject\CosineSimilarities\Cosine" + str(idx) + ".txt"
    anomaly_path = "C:\DiplomaProject\AnomalyScores\AnomalyScores" + str(idx) + ".txt"
    likelyHood_path = "C:\DiplomaProject\LikelyHoodScores\LikelyHoodScores" + str(idx) + ".txt"
    #likelyHood_path = "C:\DiplomaProject\LikelyHoodScores\LikelyHoodScoresWindow" + str(idx) + ".txt"
    f_path = "C:\DiplomaProject\CleanFingerprints\Fingerprints" + str(idx) + ".txt"
    index_path="C:\DiplomaProject\AlmarimiIndices\Indices" + str(idx) + ".txt"
    SyntaxIds_path="C:\DiplomaProject\SyntaxIds\Ids" + str(idx) + ".txt"
    SemanticIds_path = "C:\DiplomaProject\SemanticIds\Ids" + str(idx) + ".txt"
    SyntaxIds=Fileparser.get_indices_from_file(SyntaxIds_path)
    SemanticIds=Fileparser.get_indices_from_file(SemanticIds_path)
    originalCorpus=Fileparser.get_corpus_from_file(originalCorpusPath)
    cosineDoc2Vec = list(map(float, Fileparser.get_indices_from_file(cosPath)))
    anomalies=list(map(float,Fileparser.get_indices_from_file(anomaly_path)))
    likelyhood=list(map(float,Fileparser.get_indices_from_file(likelyHood_path)))
    indices=list(map(float,Fileparser.get_indices_from_file(index_path)))
    euclidean,cosine,jaccard=RetinaOperations.readSimilarities(idx)
    euclidean=list(map(float,euclidean))
    cosine = list(map(float, cosine))
    jaccard = list(map(float, jaccard))
    meanFreq,lowestFreq,upperFreq= SyntaxParser.calculateRelationalFrequencyOfSentence(originalCorpus)
    meanFreq = list(map(float, meanFreq))
    lowestFreq = list(map(float, lowestFreq))
    upperFreq = list(map(float, upperFreq))
    authorStyleVals=SyntaxParser.authorStyleOfSentences(originalCorpus)
    cosineScores=[]
    mean=[]
    lowest=[]
    upper=[]
    authorStyle=[]
    for i in range(len(SyntaxIds)):
        cosineScores.append(cosineDoc2Vec[int(SyntaxIds[i])-1])
        mean.append(meanFreq[int(SyntaxIds[i])-1])
        lowest.append(lowestFreq[int(SyntaxIds[i])-1])
        upper.append(upperFreq[int(SyntaxIds[i])-1])
        authorStyle.append(authorStyleVals[int(SyntaxIds[i])-1])
    anomaly=[]
    likely=[]
    euclid=[]
    cos=[]
    jacc=[]

    for i in range(len(SemanticIds)):
        anomaly.append(anomalies[int(SemanticIds[i])-1])
        likely.append(likelyhood[int(SemanticIds[i])-1])
        euclid.append(euclidean[int(SemanticIds[i])-1])
        cos.append(cosine[int(SemanticIds[i])-1])
        jacc.append(jaccard[int(SemanticIds[i])-1])
    metricList = []
    for i in range(len(sentences)):
        metric=OffsetMetrics(i, end[i], anomaly[i], likely[i], cosineScores[i], euclid[i], cos[i], jacc[i], mean[i], lowest[i], upper[i], authorStyle[i])
        metricList.append(metric)


    #cosineD2V_norm =normalize_data(cosineDoc2Vec)
    #anomalies_norm=normalize_data(anomalies)
    #likelyhood_norm = normalize_data(likelyhood)
    #euclidean_norm = normalize_data(euclidean)
    #cosine_norm = normalize_data(cosine)
    #jaccard_norm = normalize_data(jaccard)
    #metricList=[]
    #normList=[]
    #for i in range(len(corpus)):
        #metric=CorpusMetrics(i,indices[i],anomalies[i],likelyhood[i],corpus[i],cosineDoc2Vec[i],euclidean[i],cosine[i],jaccard[i],
                             #euclidean_norm[i]-cosine_norm[i]-jaccard_norm[i])
        #metricList.append(metric)
        #metric.toString()
        #score=likelyhood_norm[i]-cosineD2V_norm[i]+euclidean_norm[i]-cosine_norm[i]-jaccard_norm[i]
        #normMetric=CorpusMetrics(i,indices[i],anomalies_norm[i],likelyhood_norm[i],corpus[i],cosineD2V_norm[i],euclidean_norm[i],cosine_norm[i],jaccard_norm[i],score)
        #normList.append(normMetric)


    return metricList

def normalize_data(data):
    # normalizes the input features
    minVal=min(data)
    maxVal=max(data)
    normVals=[]
    for d in data:
        normVals.append((d-minVal)/(maxVal-minVal))
    return normVals

def returnTop(normalizedMetrics,number):
    # returns the top values of the normalized metrics
    sorted_metrics = sorted(normalizedMetrics, key=attrgetter('score'), reverse=True)
    for i in range(number):
        sorted_metrics[i].toString()

def filterByD2VAndLikelyhood(metricList):
    # filtering of results by doc2vec and likelihood - deprecated
    newList=[]
    indices=[]
    for m in metricList:
        if (m.likelyhoodScore>0.7 and m.score>0.4 and m.cosineDoc2Vec < 0):
        #if (m.likelyhoodScore > 0.9 and m.score > 0 or m.cosineDoc2Vec < 0):
            newList.append(m)
            indices.append(m.id)
            #m.toString()
    return newList, indices

def compareAnomalousTexts(metricList,indices,idx):
    # creates a score to separate non-anomalous parts from anomalous
    f_path = "C:\DiplomaProject\CleanFingerprints\Fingerprints" + str(idx) + ".txt"
    fingerprints = Fileparser.get_clean_fingerprints_from_file(f_path)
    euclidean,cosine,jaccard=RetinaOperations.calculateOverlapScoresOfParts(fingerprints, indices)
    euclidean=normalize_data(euclidean)
    cosine=normalize_data(cosine)
    jaccard=normalize_data(jaccard)
    for i in range(len(metricList)):
        metricList[i].score=euclidean[i]-cosine[i]-jaccard[i]

        metricList[i].toString()
    return metricList

def tagPlagiarisms(idx):
    # creates a list of detections from the predictions of parts of a document
    filename = "C:\DiplomaProject\PredictionResults\Results"
    fPath=filename+str(idx)+".txt"
    partname="C:\DiplomaProject\partIds\partIds" + str(idx) + ".txt"
    resultVals=Fileparser.get_indices_from_file(fPath)
    plagiarismList = []
    path = "C:\DiplomaProject\AlmarimiDocuments"
    documentList = Fileparser.extract_plagiarisms_from_files(path)
    text = Fileparser.extract_text_from_document(path, documentList[idx])
    start, end, sentences = Fileparser.split_into_sentences(text)
    if(len(resultVals)==0):
        return plagiarismList
    startIds,endIds=Fileparser.get_partIds_from_file(partname)

    parts=[]
    for i in range(len(resultVals)):
        if(float(resultVals[i])>0.7):
            part=partStats(int(startIds[i]),int(endIds[i]),int(int(endIds[i])-int(startIds[i])+1),float(resultVals[i]))
            parts.append(part)

    longestParts = sorted(parts, key=lambda x: x.partLength, reverse=True)
    sameStarts=set()
    newParts=[]
    for p in longestParts:
        if (p.startId not in sameStarts):
            sameStarts.add(p.startId)
            newParts.append(p)
    longestParts = sorted(newParts, key=lambda x: x.partLength, reverse=True)
    sameEnds = set()
    newParts = []
    for p in longestParts:
        if (p.endId not in sameEnds):
            sameEnds.add(p.endId)
            newParts.append(p)
    indices=[]
    for p in newParts:
        pair=[p.startId,p.endId]
        indices.append(pair)
    if(len(indices)==0):
        return plagiarismList
    indices.sort(key=lambda interval: interval[0])
    merged = [indices[0]]
    for current in indices:
        previous = merged[-1]
        if current[0] <= previous[1]:
            previous[1] = max(previous[1], current[1])
        else:
            merged.append(current)



    for p in merged:


        plag=plagiarism(start[p[0]],end[p[1]]-start[p[0]]+1)
        plag.toString()
        plagiarismList.append(plag)


    return plagiarismList



def runOutlierDetector():
    # a method for evaluating the list of our detections adn outputting them to a file
    precisions = []
    recalls = []
    accuracies =[]
    fullOverlapList=[]
    partialOverlapList=[]
    numberOfDetectedPlags=[]
    for i in range(0,40):

        detectedList=tagPlagiarisms(i)
        numberOfDetectedPlags.append(len(detectedList))
        path="C:\DiplomaProject\AlmarimiDocuments"
        documentList=Fileparser.extract_plagiarisms_from_files(path)
        plagiarismList=documentList[i].plagiarismList

        text=Fileparser.extract_text_from_document(path,documentList[i])
        precision,recall,accuracy,fullDetections,partialDetections=Fileparser.confusionMatrix(text,plagiarismList,detectedList)
        precisions.append(precision)
        recalls.append(recall)
        accuracies.append(accuracy)
        fullOverlapList.append(fullDetections)
        partialOverlapList.append(partialDetections)
    print(len(precisions))
    print(len(accuracies))
    print(len(recalls))

    filepath ="C:\DiplomaProject\OutputFile.txt"
    with open(filepath, 'w', encoding="utf-8", errors='ignore') as f:
        f.truncate(0)
        for j in range(0,40):
                f.write("id: " + str(j) +'\n')
                f.write("precision: " + str(precisions[j]) +"\n")
                f.write("recall: " + str(recalls[j])+"\n")
                f.write("accuracy: " + str(accuracies[j])+"\n")
                f.write("number of detections: " + str(numberOfDetectedPlags[j])+"\n")
                f.write("FullDetections: " + str(fullOverlapList[j])+"\n")
                f.write("PartialDetections: " + str(partialOverlapList[j])+"\n")

                f.write("----------------\n")
        f.close()



