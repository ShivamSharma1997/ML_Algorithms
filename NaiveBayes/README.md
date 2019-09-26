						ASSIGNMENT 1

					Naive Bayes Algorithm


There are four scripts in this folder, code, naiveBayes, evalScript and evalLib.

I. code Script:

The code script deals with the importing and processing of data as well as training of model and prediction and saving of results. 

Instructions to run the code script:

1. Input data file name and the label file name in the terminal. So if your data file name is ‘dataFile’ and label file name is ‘labelFile’ run the following code on the terminal:
			python code.py dataFile labelFile

2. If you want to save the predictions to a specific file, say predictionFile, run the following code in the terminal:
			python code.py dataFile labelFile -save predictionFile


II. naiveBayes Script:

The naiveBayes script contains the class, NaiveBayes, to train model and predict labels using the Naive Bayes Algorithm.

Insturctions to run the naiveBayes script:

1. Import the NaiveBayes class to your main code script.

2. Initialize the class.

3. Train the class variable using the train() function.

4. Predict the results using the predction() function.

III. evalSript Script:

The evalScript script evaluates different scores for the input truth file and prediction file.

There are three scores to check from:

1. To get the accuracy score use the following command in the terminal:
			python evalScript.py truthFile predictionFile -accuracy
2. To get the balanced error score use the following command in the terminal:
			python evalScript.py truthFile predictionFile -balanced_error
3. To get the balanced accuracy score use the following command in the terminal:
			python evalScript.py truthFile predictionFile -balanced_score

The script supports multiple score printing, so user can print multiple scores at same time.

IV. evalLib Script:

The evalLib script contains the class evaluation which calculates different scores for given truth and prediction values.

Instructions to run the evalLib Script:

1. Call evaluation class to main scipt.

2. Initiate class.

3. The functions, accuracy_score(), balanced_error_score(), balanced_accuracy_score(), return accuracy, balanced error and balanced score, respectively.
