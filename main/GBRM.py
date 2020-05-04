import main.Fileparser as Fileparser
import numpy as np
import math as math
import main.OutlierDetector as OutlierDetector
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import pickle

def offsetLossScores(idx):
    # calculates an alternative way of loss where the loss score of a sentence index would be calculated in relation
    # to the distance of the beginning and end of plagiarisms - deprecated
    indices,sentCount=getPlagIndices(idx)
    plagMaxDist=[indices[0]-0]
    for i in range(1,len(indices)):
        plagMaxDist.append(math.ceil((indices[i]-indices[i-1])/2))
        plagMaxDist.append(math.ceil((indices[i] - indices[i - 1]) / 2))
    plagMaxDist.append(sentCount-1-indices[len(indices)-1])

    plagDist=[]
    for i in range(sentCount):
        min = math.inf
        id= 0
        for j in range(len(indices)):
            dist=indices[j]-i
            if (abs(dist)<min):
                min = dist
                id = j*2
                if (dist < 0):
                   min = -1*min
                   id+=1

        plagDist.append(min/plagMaxDist[id])
        print(plagDist[i])


    return plagDist

def getPlagIndices(idx):
    # returns the beginning and ending indices of plagiarisms
    path = "C:\DiplomaProject\AlmarimiDocuments"
    documentList = Fileparser.extract_plagiarisms_from_files(path)
    plagiarismList = documentList[idx].plagiarismList
    text = Fileparser.extract_text_from_document(path, documentList[idx])
    start, end, sents = Fileparser.split_into_sentences(text)
    plagIndices = []
    if (len(plagiarismList) == 0):
        return plagIndices,len(end)

    for p in plagiarismList:
        plagIndices.append(p.offset)
        plagIndices.append(p.offset + p.length)
    pId = 0
    indices = []
    for i in range(len(plagIndices)):
        while (end[pId] < plagIndices[i] - 1):
            pId += 1
        indices.append(pId)
    return indices,len(end)


def partLossScores(startId,endId,idx):
    # a method of calculating the loss scores of parts by precision - deprecated
    indices, sentCount = getPlagIndices(idx)
    plagStarts=[]
    plagEnds=[]
    for i in range(len(indices)/2):
        plagStarts.append(indices[i*2])
        plagEnds.append(indices[i*2+1])
    hitCount=0
    for i in range(startId,endId+1):
        for j in range(len(plagStarts)):
            if (plagStarts[j]<=i<=plagEnds[j]):
                hitCount+=1
    precision=hitCount/(endId-startId+1)



    return precision

def writeOffsetLoss():
    # writes the loss of the classifier into a file
    path = "C:\DiplomaProject\AlmarimiDocuments"
    documentList = Fileparser.extract_plagiarisms_from_files(path)
    lossPath = "C:\DiplomaProject\offsetLoss\Loss"
    for i in range(len(documentList)):
        losses=offsetLossScores(i)
        lPath=lossPath+str(i) + ".txt"
        with open(lPath, 'w', encoding="utf-8", errors='ignore') as f:
            f.truncate(0)
            for j in range(len(losses)):
                f.write(str(losses[j]))
                f.write("|")
            f.close()


def prepareOffsetDataset(indexList):
    # creates the input and label data for the classifier from a list of indices
    sampleData=[]
    labels=[]
    for i in indexList:
        metrics=OutlierDetector.getOffsetCorpusMetrics(i)
        data=offsetDataFromMetricList(metrics)
        sampleData+=data

        plagIndices,length = getPlagIndices(i)
        labelList=np.zeros(length)
        for p in plagIndices:
            labelList[p]=1
        labels.extend(labelList)
    return sampleData,labels

def partsFromPrediction(prediction,idx):
    # creates potential parts from the output of the GBRT classifier
    potentialIndices=[]
    sentenceLength=len(prediction)
    for i in range(len(prediction)):
        if (prediction[i]==1):
            potentialIndices.append(i)
    startIds=[]
    endIds=[]
    if (len(potentialIndices)==0):
        return startIds,endIds
    unPairedIndices=set()
    distances=[]
    if(len(potentialIndices)==1):
        unPairedIndices.add(potentialIndices[0])
    else:
        for i in range(1,len(potentialIndices)):
            distances.append(potentialIndices[i]-potentialIndices[i-1])
            unPairedIndices.add(potentialIndices[i-1])
        unPairedIndices.add(potentialIndices[len(potentialIndices)-1])
        for i in range(len(distances)):
            if(distances[i]<=50):
                startIds.append(potentialIndices[i])
                endIds.append(potentialIndices[i+1])
                if(potentialIndices[i] in unPairedIndices):
                    unPairedIndices.remove(potentialIndices[i])
                if(potentialIndices[i+1] in unPairedIndices):
                    unPairedIndices.remove(potentialIndices[i+1])

    starts,ends=idsFromUnpairedIndices(unPairedIndices,sentenceLength)
    if(len(startIds)>0):
        if(len(starts)>0):
            startIds.extend(starts)
            endIds.extend(ends)
    else:
        if(len(starts)>0):
            startIds=starts
            endIds=ends
    return startIds,endIds

def preparePartLabels(startIds,endIds,idx):
    # prepares the labels of part
    plagIndices,length=getPlagIndices(idx)
    print(len(plagIndices))

    print(len(endIds))
    labelScores=[]
    for i in range(len(startIds)):
        overlap=0
        for j in range(int(len(plagIndices)/2)):
            overlap+=max(0, min(endIds[i], plagIndices[j*2+1]) - max(startIds[i], plagIndices[j*2])+1)
        score=overlap/(endIds[i]-startIds[i]+1)
        labelScores.append(score)
    return labelScores






def idsFromUnpairedIndices(unPairedIndices,sentenceLength):
    # creates potential parts from unpaired indices
    startIds=[]
    endIds=[]
    for idx in unPairedIndices:
        for j in range(5,50,5):
            if(idx-j>=0):
                startIds.append(idx-j)
                endIds.append(idx)
            if(idx+j<sentenceLength):
                startIds.append(idx)
                endIds.append(idx+j)
    return startIds,endIds


def preparePartDataset(indexList, predictions):
    # creates the dataset of parts used for the GBRT regressor
    sampleData = []
    labels = []
    for i in range(len(indexList)):
        startIds,endIds=partsFromPrediction(predictions[i],indexList[i])

        metrics = OutlierDetector.getPartCorpusMetrics(indexList[i],startIds,endIds)
        if(len(metrics)==0):
            continue
        data = partDataFromMetricList(metrics)
        sampleData.extend(data)
        partLabel=preparePartLabels(startIds,endIds,indexList[i])
        labels.extend(partLabel)
    return sampleData, labels


def offsetDataFromMetricList(metricList):
    # creates the input parameters for the classifier
    data = []
    for m in metricList:
        sample=[m.likelyhoodScore,m.cosineDoc2Vec,m.euclideanDistance,m.cosineSimilarity,m.jaccardIndex,m.meanFreq,m.lowestFreq,m.upperFreq,m.authorStyleVals]
        data.append(sample)
    return data

def offsetDataVector(data,id):
    #creates a vector from the input parameters of the classifier
    vector =[]
    for d in data:
        vector.append(d[id])
    return vector


def partDataFromMetricList(metricList):
    # creates the input parameters for the regressor
    data = []
    for m in metricList:
        sample=[m.cosineDoc2Vec,m.euclideanDistance, m.cosineSimilarity,
                 m.jaccardIndex,m.meanFreq,m.lowestFreq,m.upperFreq,m.meanStyle,m.partEuclid,m.partCosine,
                m.partJaccard,m.partMeanFreq,m.partLowestFreq,m.partUpperFreq]
        data.append(sample)
    return data



def readOffsetLoss(idx):
    # reads the loss scores created for the regressor - redacted
    lossPath = "C:\DiplomaProject\offsetLoss\Loss"+str(idx)+".txt"
    strLosses = Fileparser.get_indices_from_file(lossPath)
    losses = [float(i) for i in strLosses]
    return losses

def trainGBRTOffset(trainIndexList):
    # trains the Gradient Boosting classifier for detecting input anomalies
    x_train,y_train = prepareOffsetDataset(trainIndexList)
    GBRT = GradientBoostingClassifier(n_estimators=200, max_depth=4)
    GBRT.fit(x_train, y_train)
    trainingPredictions=[]
    for idx in trainIndexList:
        idList=[idx]
        x,y=prepareOffsetDataset(idList)
        prediction=GBRT.predict(x)
        trainingPredictions.append(prediction)

    #acc = GBRT.score(x_train, y_train)
    #print('trainingACC: %.4f' % acc)
    #acc = GBRT.score(x_test, y_test)
    #print('trainingACC: %.4f' % acc)
    filename ="C:\DiplomaProject\GBRTOffset.sav"
    pickle.dump(GBRT,open(filename,'wb'))
    return trainingPredictions,GBRT

def offsetVariableNames():
    # list of variable names of the classifier features - deprecated
    feature_names = np.array()
    feature_names.append("likelihood")
    feature_names.append("Doc2Vec")
    feature_names.append("euclidean")
    feature_names.append("cosine")
    feature_names.append("jaccard")
    feature_names.append("meanFreq")
    feature_names.append("lowestFreq")
    feature_names.append("highestFreq")
    feature_names.append("authorStyle")
    return feature_names

def plotOffsetModelVariableImportance():
    # plots the variable importance of the classifier
    filename = "C:\DiplomaProject\GBRTOffset.sav"
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    feature_importance = model.feature_importances_
    feature_names = ["likelihood","Doc2Vec", "euclidean", "cosine", "jaccard", "meanFreq", "lowestFreq", "highestFreq",
                     "authorStyle"]
    relative_importance = 100.0 * (feature_importance / feature_importance.max())

    fig = plt.figure(figsize=(7, 4))

    bar_plot = sns.barplot(x=feature_importance, y=feature_names,orient="h")
    plt.title('Feature Importance of Gradient Boosting Classifier')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')
    plt.show()

def plotPartModelVariableImportance():
    # plots the variable importance of the regressor
    filename = "C:\DiplomaProject\GBRTPart.sav"
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    feature_names=["Doc2Vec","euclidean","cosine","jaccard","meanFreq","lowestFreq","highestFreq","authorStyle","partEuc","partCos","partJac","partMean","partLowest","partHighest"]
    feature_importance = model.feature_importances_
    relative_importance = 100.0 * (feature_importance / feature_importance.max())
    fig = plt.figure(figsize=(7, 4))

    bar_plot = sns.barplot(x=feature_importance, y=feature_names, orient="h")
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')
    plt.title('Feature Importance of Gradient Boosting Regressor')
    plt.show()

def plagVector(plagIndices,length,barHeight):
    # creates a vector of plagiarisms used for plotting
    vector = np.zeros(length)
    for i in range(int(len(plagIndices)/2)):
        for j in range(plagIndices[i*2],plagIndices[i*2+1]+1):
            vector[j]=barHeight
    return vector

def plotIndividualMetrics(id,metricNum):
    # creates a plot that examines how one of the features of the regressor changes for each sentence in comparison to the location
    # of the plagiarisms
    metrics = OutlierDetector.getOffsetCorpusMetrics(id)
    data = offsetDataFromMetricList(metrics)
    dataVector = offsetDataVector(data,metricNum)
    dataVector = OutlierDetector.normalize_data(dataVector)
    dataHeight=max(dataVector)
    print(dataHeight)
    barHeight=dataHeight + dataHeight/4
    plagIndices,length = getPlagIndices(id)
    xvals =[]
    for i in range(0,length):
        xvals.append(i)
    pVector = plagVector(plagIndices,length,barHeight)
    fig = plt.figure()
    fig, ax = plt.subplots(figsize=(15, 6))

    bar_plot = sns.barplot(x=xvals, y=pVector, ax=ax, color = "r")
    bar_plot.set(xticks=[])
    ax2 = ax.twinx()

    bar_plot2 = sns.barplot(x=xvals, y=dataVector, ax=ax2, color = "b")
    bar_plot2.set(xticks=[])
    plt.show()


def predictGBRTOffset(model,testIdx):
    # predicts the potential indices of change from the trained classifier
    x_test, y_test = prepareOffsetDataset(testIdx)
    prediction=[model.predict(x_test)]
    return prediction

def trainGBRTPart(trainIndexList,trainingPrediction):
    # trains the Gradient boosting regressor to predict the relevance/precision of potential anomalous parts
    x_train, y_train = preparePartDataset(trainIndexList,trainingPrediction)
    GBRT = GradientBoostingRegressor(n_estimators=200, max_depth=4)
    GBRT.fit(x_train, y_train)
    filename ="C:\DiplomaProject\GBRTPart.sav"
    pickle.dump(GBRT, open(filename, 'wb'))
    return GBRT

def predictGBRTPart(testIndexList,testPrediction,model):
    # predicts the relevance/precision of potential anomalous parts of change from the trained regressor
    x_train, y_train = preparePartDataset(testIndexList,testPrediction)
    prediction=[]
    if(len(x_train)==0):
        return prediction
    prediction=model.predict(x_train)
    return prediction

def runPrediction():
    # runs the training and prediction of both the classifier and regressor on the entire dataset.
    # for each document returns the prediction of parts
    resultname="C:\DiplomaProject\PredictionResults\Results"
    partname="C:\DiplomaProject\partIds\partIds"
    trainList=[]
    for i in range(0,40):
        trainList.append(i)

    trainingPredictions,GBRTOffset=trainGBRTOffset(trainList)
    GBRTPart=trainGBRTPart(trainList,trainingPredictions)
    for i in range(len(trainList)):
        testIdx=[trainList[i]]
        OffsetPred=predictGBRTOffset(GBRTOffset,testIdx)
        startIds,endIds=partsFromPrediction(OffsetPred[0],testIdx)
        prediction=predictGBRTPart(testIdx,OffsetPred,GBRTPart)
        fPath=resultname+str(i)+".txt"
        with open(fPath, 'w', encoding="utf-8", errors='ignore') as f:
            f.truncate(0)
            for j in range(len(prediction)):
                f.write(str(prediction[j]))
                f.write("|")
            f.close()
        idPath=partname+str(i)+".txt"
        with open(idPath, 'w', encoding="utf-8", errors='ignore') as f:
            f.truncate(0)
            for j in range(len(startIds)):
                f.write(str(startIds[j]))
                f.write(",")
                f.write(str(endIds[j]))
                f.write("|")
            f.close()


plotIndividualMetrics(17,2)