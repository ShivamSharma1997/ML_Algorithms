import sys

from evalLib import evaluation

########################## GETTING FILE PATHS ##########################

truthFile = sys.argv[1]
predFile = sys.argv[2]

########################## EXTRACTING VALUES ##########################

truthVals = open(truthFile).readlines()
truthVals = [val.strip().split() for val in truthVals]
yTrue = [int(val[0]) for val in truthVals]

predVals = open(predFile).readlines()
predVals = [val.strip().split() for val in predVals]
yPred = [int(val[0]) for val in predVals]

########################## CALCULATING AND PRINTING SCORES ##########################

evl = evaluation(yTrue, yPred)
score = {}

if '-accuracy' in sys.argv:
	score['Accuracy'] = evl.accuracy_score()
    
if '-balanced_error' in sys.argv:
	score['Balanced Error'] = evl.balanced_error_score()

if '-balanced_score' in sys.argv:
	score['Balanced Accuracy'] = evl.balanced_accuracy_score()

if len(score.keys()) != 0:
	for key in score.keys():
		print('\n', key, ' : ', score[key])