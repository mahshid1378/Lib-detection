import numpy as np
import SimpleNavieBayes.NavieBayes as naiveBayes

filename = 'C:/Users/hp 850/Desktop/SMSCollection.txt'
smsWords, classLables = naiveBayes.loadSMSData(filename)
vocabularyList = naiveBayes.createVocabularyList(smsWords)
print ("Generate corpus!")
trainMarkedWords = naiveBayes.setOfWordsListToVecTor(vocabularyList, smsWords)
print ("Data marking is complete!")
trainMarkedWords = np.array(trainMarkedWords)
print ("The data is converted into a matrix!")
pWordslie, pWordsTrue, pLie = naiveBayes.trainingNaiveBayes(trainMarkedWords, classLables)
print ('pLie:', pLie)
fpLie = open('C:/Users/hp 850/Desktop/pLie.txt', 'w')
Lie = pLie.__str__()
fpLie.write(Lie)
fpLie.close()
fw = open('C:/Users/hp 850/Desktop/vocabularyList.txt', 'w')
for i in range(len(vocabularyList)):
    fw.write(vocabularyList[i] + '\t')
fw.flush()
fw.close()
np.savetxt('C:/Users/hp 850/Desktop/vocabularyList.txt', pWordsSpamicity, delimiter='\t')
np.savetxt('C:/Users/hp 850/Desktop/pWordsTrue.txt', pWordsHealthy, delimiter='\t')