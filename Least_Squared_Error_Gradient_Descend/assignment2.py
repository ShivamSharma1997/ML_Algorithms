import sys

from gradientDescent import gradientDescent

################ DEFINING HYPERPARAMETERS AND FILE PATHS ################

if '-save' not in sys.argv:

    dataFile = sys.argv[1]
    labelFile = sys.argv[2]
    eta = float(sys.argv[3])
    theta = float(sys.argv[4])

elif '-save' in sys.argv:
    dataFile = sys.argv[1]
    labelFile = sys.argv[2]
    eta = float(sys.argv[3])
    theta = float(sys.argv[4])
    saveFile = sys.argv[-1]

################ DATA PRE-PROCESSING ################

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
    
    if label == 1:
        trainY.append(1)
    elif label == 0:
        trainY.append(-1)
    
    trainIndex.append(i)

testX = []
testIndex = [i for i in range(len(data)) if i not in trainIndex]

if len(testIndex) == 0:
    testIndex = trainIndex

else:
    for i in testIndex:
        testX.append(data[i])

if len(testX) == 0:
    testX = trainX

################ TRAINING MODEL ################

model = gradientDescent(eta, theta)
model.train(trainX, trainY)

print('Learned Weights:')
print(model.weights[1:])

print('\nPrediction:')
predictions = model.predict(testX)
print(predictions)

print('\nDistance from Origin:')
print(model.distToOrigin())

################ SAVING PREDICTIONS ################

if '-save' in sys.argv:
    
    wrtFile = open(saveFile, 'w')
    
    for pred, idx in zip(predictions, testIndex):
        wrtFile.writelines('{} {}\n'.format(pred, idx))
    
    wrtFile.close()