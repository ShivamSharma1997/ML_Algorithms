				EVALUATION


There are two scripts in this folder, evalScript and evalLib.

I. evalSript Script:

The evalScript script evaluates different scores for the input truth file and prediction file.

There are three scores to check from:

1. To get the accuracy score use the following command in the terminal:
			python evalScript.py truthFile predictionFile -accuracy
2. To get the balanced error score use the following command in the terminal:
			python evalScript.py truthFile predictionFile -balanced_error
3. To get the balanced accuracy score use the following command in the terminal:
			python evalScript.py truthFile predictionFile -balanced_score

The script supports multiple score printing, so user can print multiple scores at same time.

II. evalLib Script:

The evalLib script contains the class evaluation which calculates different scores for given truth and prediction values.

Instructions to run the evalLib Script:

1. Call evaluation class to main scipt.

2. Initiate class.

3. The functions, accuracy_score(), balanced_error_score(), balanced_accuracy_score(), return accuracy, balanced error and balanced score, respectively.
