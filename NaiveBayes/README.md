				Naive Bayes Algorithm


There are two scripts in this folder, code and naiveBayes.

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

