import SimpleNavieBayes.NavieBayes as naiveBayes
import random
import numpy as np
def simpleTest():
    vocabularyList, pWordslie, pWordsTrue, pLie = \
        naiveBayes.getTrainedModelInfo()
    filename = 'C:/Users/hp 850/Desktop/testing.txt'
    smsWords, classLables = naiveBayes.loadSMSData(filename)
    smsType = naiveBayes.classify(vocabularyList, pWordsTrue, pWordslie, pLie, smsWords[0])
    print (smsType)

def testClassifyErrorRate():
    filename = 'C:/Users/hp 850/Desktop/traning.txt'
    smsWords, classLables = naiveBayes.loadSMSData(filename)
    testWords = []
    testWordsType = []
    testCount = 1000
    for i in range(testCount):
        randomIndex = int(random.uniform(0, len(smsWords)))
        testWordsType.append(classLables[randomIndex])
        testWords.append(smsWords[randomIndex])
        del (smsWords[randomIndex])
        del (classLables[randomIndex])

    vocabularyList = naiveBayes.createVocabularyList(smsWords)
    print ("Generate corpus!")
    trainMarkedWords = naiveBayes.setOfWordsListToVecTor(vocabularyList, smsWords)
    print ("Data marking is complete!")
    trainMarkedWords = np.array(trainMarkedWords)
    print ("The data is converted into a matrix!")
    pWordsTrue, pWordslie, pLie = naiveBayes.trainingNaiveBayes(trainMarkedWords, classLables)
    errorCount = 0.0
    for i in range(testCount):
        smsType = naiveBayes.classify(vocabularyList, pWordsTrue, pWordslie, pLie, testWords[i])
        print ('Forecast category:', smsType, 'Actual category:', testWordsType[i])
        if smsType != testWordsType[i]:
            errorCount += 1
    print ('Number of errors:', errorCount, 'Error rate:', errorCount / testCount)

if __name__ == '__main__':
    testClassifyErrorRate()