import os
import re
import difflib
import nltk
from io import open
import sys
import glob
import xml.dom.minidom
from collections import namedtuple
import xml.etree.ElementTree as ET
import re
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from gensim import utils
import gensim.parsing.preprocessing as gsp
from nltk.stem import PorterStemmer
import numpy as np


class document:
    # A class for storing a document and its corresponding list of plagiarism class objects
  def __init__(self, name, plagiarismList):
    self.name = name
    self.plagiarismList = plagiarismList

  def toString(self):
    print(self.name)
    for p in self.plagiarismList:
        string = p.toString()
        print("plagiarism: " + string)

class plagiarism:
    # A class of the plagiarism object, defined by lenght and offset
    def __init__(self, offset, length):
        self.offset = offset
        self.length = length

    def toString(self):
        string = "offset: " + str(self.offset) + " length: " + str(self.length)
        return string

ps=PorterStemmer()
  # list of stopwords used by the algortihm
stopWords = ([
    'all', 'six', 'just', 'less', 'being', 'indeed', 'over', 'move', 'anyway', 'four', 'not', 'own', 'through',
    'using', 'fifty', 'where', 'mill', 'only', 'find', 'before', 'one', 'whose', 'system', 'how', 'somewhere',
    'much', 'thick', 'show', 'had', 'enough', 'should', 'to', 'must', 'whom', 'seeming', 'yourselves', 'under',
    'ours', 'two', 'has', 'might', 'thereafter', 'latterly', 'do', 'them', 'his', 'around', 'than', 'get', 'very',
    'de', 'none', 'cannot', 'every', 'un', 'they', 'front', 'during', 'thus', 'now', 'him', 'nor', 'name', 'regarding',
    'several', 'hereafter', 'did', 'always', 'who', 'didn', 'whither', 'this', 'someone', 'either', 'each', 'become',
    'thereupon', 'sometime', 'side', 'towards', 'therein', 'twelve', 'because', 'often', 'ten', 'our', 'doing', 'km',
    'eg', 'some', 'back', 'used', 'up', 'go', 'namely', 'computer', 'are', 'further', 'beyond', 'ourselves', 'yet',
    'out', 'even', 'will', 'what', 'still', 'for', 'bottom', 'mine', 'since', 'please', 'forty', 'per', 'its',
    'everything', 'behind', 'does', 'various', 'above', 'between', 'it', 'neither', 'seemed', 'ever', 'across', 'she',
    'somehow', 'be', 'we', 'full', 'never', 'sixty', 'however', 'here', 'otherwise', 'were', 'whereupon', 'nowhere',
    'although', 'found', 'alone', 're', 'along', 'quite', 'fifteen', 'by', 'both', 'about', 'last', 'would',
    'anything', 'via', 'many', 'could', 'thence', 'put', 'against', 'keep', 'etc', 'amount', 'became', 'ltd', 'hence',
    'onto', 'or', 'con', 'among', 'already', 'co', 'afterwards', 'formerly', 'within', 'seems', 'into', 'others',
    'while', 'whatever', 'except', 'down', 'hers', 'everyone', 'done', 'least', 'another', 'whoever', 'moreover',
    'couldnt', 'throughout', 'anyhow', 'yourself', 'three', 'from', 'her', 'few', 'together', 'top', 'there', 'due',
    'been', 'next', 'anyone', 'eleven', 'cry', 'call', 'therefore', 'interest', 'then', 'thru', 'themselves',
    'hundred', 'really', 'sincere', 'empty', 'more', 'himself', 'elsewhere', 'mostly', 'on', 'fire', 'am', 'becoming',
    'hereby', 'amongst', 'else', 'part', 'everywhere', 'too', 'kg', 'herself', 'former', 'those', 'he', 'me', 'myself',
    'made', 'twenty', 'these', 'was', 'bill', 'cant', 'us', 'until', 'besides', 'nevertheless', 'below', 'anywhere',
    'nine', 'can', 'whether', 'of', 'your', 'toward', 'my', 'say', 'something', 'and', 'whereafter', 'whenever',
    'give', 'almost', 'wherever', 'is', 'describe', 'beforehand', 'herein', 'doesn', 'an', 'as', 'itself', 'at',
    'have', 'in', 'seem', 'whence', 'ie', 'any', 'fill', 'again', 'hasnt', 'inc', 'thereby', 'thin', 'no', 'perhaps',
    'latter', 'meanwhile', 'when', 'detail', 'same', 'wherein', 'beside', 'also', 'that', 'other', 'take', 'which',
    'becomes', 'you', 'if', 'nobody', 'unless', 'whereas', 'see', 'though', 'may', 'after', 'upon', 'most', 'hereupon',
    'eight', 'but', 'serious', 'nothing', 'such', 'why', 'off', 'a', 'don', 'whereby', 'third', 'i', 'whole', 'noone',
    'sometimes', 'well', 'amoungst', 'yours', 'their', 'rather', 'without', 'so', 'five', 'the', 'first', 'with',
    'make', 'once'
])


def get_clean_fingerprints_from_file(path):
    # returns clean fingerprints from a file
    printList = []
    with open(path, encoding="utf-8") as f:
        text = f.read()
        fingerprints = text.split("|")
        for i in range(len(fingerprints) - 1):
            fPrint = fingerprints[i].split(",")
            fPrint.pop()
            printList.append(fPrint)
        f.close()
    return printList



def extract_text_from_document(path, document):
    # extracts raw text from documents
    filename = str(path) + "/" + str(document.name)
    with open(filename, encoding="utf-8", errors='ignore') as f:
        text = f.read()
    return str(text)

def extract_plagiarisms_from_text(text, document):
    # extracts list of plagiarisms from a document
    plagiarisms = [];
    for p in document.plagiarismList:
            start = p.offset
            end  = start + p.length
            plagiarisms.append(text[start:end])
    return plagiarisms


# Methods for extracting plagiarisms from XML files - from PAN 2011
def extract_plagiarisms_from_files(path):
    """Returns a set of plagiarism annotations from XML files below path."""
    if not os.path.exists(path):
        print("Path not accessible:", path)
        sys.exit(2)
    documentList = [];
    num_files = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    for i in range(1,int(num_files/2)+1):
        idxLen = len(str(i));
        filename = "suspicious-document0"
        for j in range(4-idxLen):
            filename += "0"
        filename+=str(i)
        docName=filename+".txt"
        xmlName=path+"/"+filename+".xml"
        plagiarismList = extract_plagiarisms_from_file(xmlName)
        documentList.append(document(docName,plagiarismList))
    return documentList


def extract_plagiarisms_from_file(xmlfile):
    """Returns a set of plagiarism annotations from an XML file."""
    doc = xml.dom.minidom.parse(xmlfile)
    plagiarismList = []
    for node in doc.documentElement.childNodes:
        if node.nodeType == xml.dom.Node.ELEMENT_NODE and \
           node.hasAttribute('name') and \
           node.getAttribute('name').endswith("plagiarism"):
            plagiarism = extract_plagiarism_from_node(node)
            if plagiarism:
                plagiarismList.append(plagiarism)
    return plagiarismList


def extract_plagiarism_from_node(xmlnode, ):
    """Returns a plagiarism annotation from an XML feature tag node."""
    if not (xmlnode.hasAttribute('this_offset') and \
            xmlnode.hasAttribute('this_length')):
        return False
    offset = int(xmlnode.getAttribute('this_offset'))
    length = int(xmlnode.getAttribute('this_length'))

    return plagiarism(offset, length)





def findOverlaps(plagiarismList, detectionList):
    # method that finds the number of overlapping characters within our dataset,
    # and looks for fully overlapping and partially overlapping sections
    overlap =0
    partialDetections=0
    fullyDetected=0
    fullyRelevant=0
    for detection in detectionList:
        for plagiarism in plagiarismList:
            overlapSize= max(0, min(detection.offset+detection.length, plagiarism.offset+plagiarism.length) -
                             max(detection.offset, plagiarism.offset))
            overlap+=overlapSize
            if(overlapSize>0):
                if(overlapSize==plagiarism.length):
                    fullyDetected+=1
                else:
                    partialDetections+=1
    return overlap, fullyDetected, partialDetections

def listLength(list):
    # calculates the sum of the lengths of the individual elements in the list
    counter=0
    for l in list:
        counter+=l.length
    return counter

def confusionMatrix(text,plagiarismList, detectionList):
    # calculates precision, recall, accuracy and the overlaps of the detections and actual plagiarisms
    fullLength=len(text)
    plagiarismLength=listLength(plagiarismList)

    detectionLength=listLength(detectionList)



    TPLength,fullDetections,partialDetections=findOverlaps(plagiarismList,detectionList)
    FPLength=detectionLength-TPLength
    FNLength=plagiarismLength-TPLength
    TNLength=fullLength-TPLength-FPLength-FNLength
    if (detectionLength==0):
        precision=-1
    else:
        precision=TPLength/detectionLength
    if (plagiarismLength == 0):
        recall=-1
    else:
        recall = TPLength / plagiarismLength

    accuracy=(TPLength+TNLength)/fullLength
    print(precision)
    print(recall)
    print(accuracy)
    return precision,recall,accuracy,fullDetections,partialDetections


def split_into_sentences(text):
    # splits the text into sentences and also preserves the corresponding starting and ending indices
    startIndices=[]
    endIndices=[]
    corpus=[]
    punkt_param = PunktParameters()
    punkt_param.abbrev_types = set(['dr', 'doc', 'mr', 'mrs', 'prof', 'inc', 'mgr', 'ing', 'st'])
    sentence_splitter = PunktSentenceTokenizer(punkt_param)

    for start, end in sentence_splitter.span_tokenize(text):
        startIndices.append(start)
        endIndices.append(end)
        token = text[start:end]
        corpus.append(token)
    return startIndices, endIndices, corpus

# methods for cleaning the text of non-alphabet characters
def clean_corpus(corpus):
    for i in range(len(corpus)):
        corpus[i]=clean_text(corpus[i]);
    return corpus

def prepare_corpus(corpus):
    # only removes non-alphabet characters
    for i in range(len(corpus)):
        text=corpus[i]
        text = text.lower()
        text = re.sub("[^a-zA-Z'\d\s:]", " ", text)
        text = re.sub(r'\s+', ' ', text)
        text = str(text)
        corpus[i]=text
    return corpus


def  clean_text(text):
    # removes non-alphabet characters, stopwords and short-words
    text = text.lower()
    text = re.sub("[^a-zA-Z'\d\s:]", " ", text)
    text = re.sub(r'\s+', ' ', text)
    text=str(text)
    shortword = re.compile(r'\W*\b\w{1,3}\b')
    text=shortword.sub('', text)
    text = remove_stopwords(text)
    return text

def remove_stopwords(text):
    # removes stopwords from text using the list
    wordList = text.split()

    finalList = [word for word in wordList if word not in stopWords]

    text = ' '.join(finalList)
    return text

def tag_sentence_indices(corpus):
    # tags sentence indices to know what character offset they occur at - deprecated
    idxTags = [];
    offset=0;
    for i in range(len(corpus)):
        idxTags.append(offset);
        offset+=len(corpus[i]);
    return idxTags;

def merge_corpus(corpus, endIndices):
    # merges the corpus so that sentences containing a single word will get merged with the following ones.
    # we will save the ending indices so we know where each new sentence ends and which merged sentence each sentence belongs to
    newCorpus=[]
    newIndices=[]
    idxList=[]
    buffer="";
    bufferCount=0
    counter=0
    for i in range(len(corpus)-1):
        if len(corpus[i].split()) > 1:
            counter+=1
            newCorpus.append(str(buffer + " " + corpus[i]))
            newIndices.append(endIndices[i])
            idxList.append(counter)
            for j in range(bufferCount):
                idxList.append(counter)
            buffer = ""
            bufferCount=0
        else:
            buffer+=str(corpus[i])
            bufferCount+=1
    finalIdx=len(corpus)-1
    finalText=str(buffer+corpus[finalIdx])
    idxList.append(counter+1)
    if (finalText.isspace()==False):
        newCorpus.append(str(finalText))
        newIndices.append(endIndices[finalIdx])
        for j in range(bufferCount):
            idxList.append(counter+1)


    return newCorpus,newIndices, idxList

def prepareDocumentDataset(documentList, sourcePath):
    # writes the split and merged sentences as well as their corresponding ending indices into a file, this version does not remove
    # stopwords and short words in addition to non-alphabet characters
    corpusPath="C:\DiplomaProject\OriginalCorpus\Corpus"
    indexPath="C:\DiplomaProject\OriginalIndices\Indices"
    idxPath="C:\DiplomaProject\SyntaxIds\Ids"
    for i in range(len(documentList)):
        text = extract_text_from_document(sourcePath, documentList[i])
        start, end, corpus= split_into_sentences(text)
        corpus = prepare_corpus(corpus)
        corpus, endIndices,idxList = merge_corpus(corpus, end)
        cPath = corpusPath+ str(i) + ".txt"
        iPAth = indexPath + str(i) + ".txt"
        idPath = idxPath + str(i) + ".txt"
        with open(cPath, 'w', encoding="utf-8", errors='ignore') as f:
            f.truncate(0)
            for j in range(len(corpus)):
                f.write(str(corpus[j]))
                f.write("|")
            f.close()
        with open(iPAth, 'w', encoding="utf-8", errors='ignore') as f:
            f.truncate(0)
            for j in range(len(endIndices)):
                f.write(str(endIndices[j]))
                f.write("|")
            f.close()
        with open(idPath, 'w', encoding="utf-8", errors='ignore') as f:
            f.truncate(0)
            for j in range(len(idxList)):
                f.write(str(idxList[j]))
                f.write("|")
            f.close()

    return corpus, endIndices,idxList



def extract_sentences_into_files(documentList, sourcePath):
    # writes the split and merged sentences as well as their corresponding ending indices into a file, this version removes
    # stopwords and short words in addition to non-alphabet characters
    corpusPath = "C:\DiplomaProject\AlmarimiCorpus"
    indexPath = "C:\DiplomaProject\AlmarimiIndices"
    idxPath = "C:\DiplomaProject\SemanticIds\Ids"
    for i in range(len(documentList)):
        text = extract_text_from_document(sourcePath, documentList[i])
        start, end, corpus= split_into_sentences(text)
        corpus=clean_corpus(corpus)
        corpus, endIndices,idxList =merge_corpus(corpus,end)

        cPath=corpusPath+"/Corpus"+str(i)+".txt"
        iPAth=indexPath+"/Indices"+str(i)+".txt"
        idPath = idxPath + str(i) + ".txt"
        with open(cPath, 'w', encoding="utf-8", errors='ignore') as f:
            f.truncate(0)
            for j in range(len(corpus)):
                f.write(str(corpus[j]))
                f.write("|")
            f.close()
        with open(iPAth, 'w', encoding="utf-8", errors='ignore') as f:
            f.truncate(0)
            for j in range(len(endIndices)):
                f.write(str(endIndices[j]))
                f.write("|")
            f.close()
        with open(idPath, 'w', encoding="utf-8", errors='ignore') as f:
            f.truncate(0)
            for j in range(len(idxList)):
                f.write(str(idxList[j]))
                f.write("|")
            f.close()

    return corpus, endIndices,idxList

def get_corpus_from_file(path):
    # returns a list of split sentences corresponding to a document
    with open(path, encoding="utf-8", errors='ignore') as f:
        text=f.read()
        corpus = text.split("|")
        f.close()
    return corpus

def get_fingerprint_from_file(path):
    # returns the list of semantic fingerprints from a file
    printList=[]
    with open(path, encoding="utf-8", errors='ignore') as f:
        text = f.read()
        fingerprints = text.split("|")
        for i in range(len(fingerprints)-1):
            print = fingerprints[i].split(",")
            print.pop()
            printList.append(print)
        f.close()
    return printList

def get_indices_from_file(path):
    # returns the list of ending indices from a file
    with open(path, encoding="utf-8", errors='ignore') as f:
        text = f.read()
        indices = text.split("|")
        indices.pop()
        f.close()
    return indices

def get_partIds_from_file(path):
    # returns the starts and ends of individual sentences from a file
    with open(path, encoding="utf-8", errors='ignore') as f:
        text = f.read()
        indices = text.split("|")
        indices.pop()
        startIds=[]
        endIds=[]
        for i in range(len(indices)):
            ids=indices[i].split(",")
            startIds.append(ids[0])
            endIds.append(ids[1])
        f.close()
    return startIds,endIds

def stemCorpus(corpus):
    # stems the words in a corpus - deprecated
    stemCorpus=[]
    for sent in corpus:
        wordList = sent.split()
        stemWords=[]
        for w in wordList:
            stemWords.append(ps.stem(w))
        stemSent = ' '.join(stemWords)
        stemCorpus.append(stemSent)
    return stemCorpus


def labelDataset(documentList):
    # labels the plagiarized part of the dataset as a positive example and the rest as negative - deprecated
    indexPath = "C:\DiplomaProject\OriginalIndices\Indices"
    idxPath = "C:\DiplomaProject\SyntaxIds\Ids"
    tagPath="C:\DiplomaProject\AnomalyTags\Tags"
    for i in range(len(documentList)):
        iPAth = indexPath + str(i) + ".txt"
        indices=get_indices_from_file(iPAth)
        idPath = idxPath + str(i) + ".txt"
        ids=get_indices_from_file(idPath)
        tPath= tagPath + str(i) + ".txt"
        plagiarisms=documentList[i].plagiarismList
        tagList=np.zeros(len(ids))
        idx=0
        for plag in plagiarisms:
            while (int(indices[idx])<plag.offset-1):
                idx+=1
            while(int(indices[idx])<=plag.offset-1+plag.length):
                tagList[int(ids[idx])]=1
                idx+=1
        with open(tPath, 'w', encoding="utf-8", errors='ignore') as f:
            f.truncate(0)
            for j in range(len(tagList)):
                f.write(str(tagList[j]))
                f.write("|")
            f.close()







#path = "C:\DiplomaProject\AlmarimiDocuments"
#documentList=extract_plagiarisms_from_files(path)
#prepareDocumentDataset(documentList,path)