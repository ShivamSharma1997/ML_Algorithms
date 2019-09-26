import sys

from naiveBayes import NaiveBayes

################# READING FILE PATHS #################

if '-save' in sys.argv:
    dataFile = sys.argv[1]
    labelFile = sys.argv[2]
    resultFile = sys.argv[-1]

else:
    dataFile = sys.argv[1]
    labelFile = sys.argv[2]

################# READ AND PROCESS DATA #################

data = open(dataFile, encoding='utf-8').readlines()
data = [line.strip().split() for line in data]
data = [list(map(float, line)) for line in data]

trainlabels = open(labelFile, encoding='utf-8').readlines()
trainlabels = [line.strip().split(' ') for line in trainlabels]
trainlabels = [list(map(int, line)) for line in trainlabels]

trainX, trainY = [], []
trainIndex = []

for (label, i) in trainlabels:
    trainX.append(data[i])
    trainY.append(label)
    trainIndex.append(i)

testX = []
testIndex = [i for i in range(len(data)) if i not in trainIndex]

for i in testIndex:
    testX.append(data[i])

################# USING NAIVE BAYES TO PREDICT SCORE #################

clf = NaiveBayes()
clf.train(trainX, trainY)
pred = clf.prediction(testX)

if '-save' in sys.argv:
    writeFile = open(resultFile, 'w')
    
    for (idx, p) in zip(testIndex, pred):
        print(p, idx)
        writeFile.writelines('{} {}\n'.format(p, idx))
    
    writeFile.close()

else:
    for (idx, p) in zip(testIndex, pred):
        print(p, idx)