import numpy as np

def textParser(text): 
    import re
    regEx = re.compile(r'[^a-zA-Z]|\d')  
    words = regEx.split(text)
    words = [word.lower() for word in words if len(word) > 0]
    return words

def loadSMSData(fileName):
    f = open(fileName)
    classCategory = []  
    smsWords = []
    for line in f.readlines():
        linedatas = line.strip().split('\t')
        if linedatas[0] == 'lie':
            classCategory.append(0)
        elif linedatas[0] == 'true':
            classCategory.append(1)
        words = textParser(linedatas[1])
        smsWords.append(words)
    return smsWords, classCategory

def createVocabularyList(smsWords): 
    vocabularySet = set([])
    for words in smsWords:
        vocabularySet = vocabularySet | set(words)
    vocabularyList = list(vocabularySet)
    return vocabularyList

def getVocabularyList(fileName):
    fr = open(fileName)
    vocabularyList = fr.readline().strip().split('\t')
    fr.close()
    return vocabularyList

def setOfWordsToVecTor(vocabularyList, smsWords):    
    vocabMarked = [0] * len(vocabularyList)
    for smsWord in smsWords:
        if smsWord in vocabularyList:
            vocabMarked[vocabularyList.index(smsWord)] += 1
    return vocabMarked

def setOfWordsListToVecTor(vocabularyList, smsWordsList):    
    vocabMarkedList = []
    for i in range(len(smsWordsList)):
        vocabMarked = setOfWordsToVecTor(vocabularyList, smsWordsList[i])
        vocabMarkedList.append(vocabMarked)
    return vocabMarkedList

def trainingNaiveBayes(trainMarkedWords, trainCategory):
    numTrainDoc = len(trainMarkedWords)
    numWords = len(trainMarkedWords[0])
    pSpam = sum(trainCategory) / float(numTrainDoc)
    wordsInlieNum = np.ones(numWords)
    wordsInTrueNum = np.ones(numWords)
    lieWordsNum = 2.0
    TrueWordsNum = 2.0
    for i in range(0, numTrainDoc):
        if trainCategory[i] == 1:  
            WordsInlieNum += trainMarkedWords[i]
            lieWordsNum += sum(trainMarkedWords[i])  
        else:
            wordsTrueNum += trainMarkedWords[i]
            TrueWordsNum += sum(trainMarkedWords[i])

    pWordslie = np.log(WordsInlieNum / lieWordsNum)
    pWordsTrue = np.log(wordsInTrueNum / TrueWordsNum)
    return pWordsTrue, pWordslie, pLie

def getTrainedModelInfo():    
    vocabularyList = getVocabularyList('C:/Users/hp 850/Desktop/vocabularyList.txt')
    pWordsTrue = np.loadtxt('C:/Users/hp 850/Desktop/pWordsTrue.txt', delimiter='\t')
    pWordslie = np.loadtxt('C:/Users/hp 850/Desktop/pWordslie.txt', delimiter='\t')
    fr = open('C:/Users/hp 850/Desktop/pLie.txt')
    pLie = float(fr.readline().strip())
    fr.close()
    return vocabularyList, pWordsTrue, pWordslie, pLie

def classify(vocabularyList, pWordsTrue, pWordslie, pLie, testWords):    
    testWordsCount = setOfWordsToVecTor(vocabularyList, testWords)
    testWordsMarkedArray = np.array(testWordsCount)
    p1 = sum(testWordsMarkedArray * pWordsSpamicity) + np.log(pLie)
    p0 = sum(testWordsMarkedArray * pWordsHealthy) + np.log(1 - pLie)
    if p1 > p0:
        return 1
    else:
        return 0