		Gradient Descent with Least Square Means Algorithm


There are two scripts in this folder, code and gradientDescent.

I. code Script:

The code script deals with the importing and processing of data as well as training of model and prediction, printing weights, predictions and length to origin and also saving of results. 

Instructions to run the code script:

1. Input data file name, label file name, eta value and theta value in the terminal. So if your data file name is dataFile, label file name is labelFile, the learning rate is eta and stopping condition is theta run the following code on the terminal:
			
				python code.py dataFile labelFile eta theta

2. If you want to save the predictions to a specific file, say predictionFile, run the following code in the terminal:
			
				python code.py dataFile labelFile eta theta -save 			      predictionFile


II. gradientDescent Script:

The gradientDescent script contains the class, gradientDescent, to train model and predict labels using the Gradient Descent for Least Squared Error Algorithm.

Instructions to run the gradientDescent script:

1. Import the gradientDescent class to your main code script.

2. Initialize the class.

3. Train the class variable using the train() function.

4. Predict the results using the predict() function.

5. Calculate the distance from origin using the distToOrigin() function.

6. Get the weights using Class_Variable_Name.weights.
