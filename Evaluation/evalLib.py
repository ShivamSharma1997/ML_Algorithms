class evaluation:
    def __init__(self, truth, pred):
        
        '''
        input: (list, list)
        
        functions: createConfusionMatrix(),
                   accuracy_score(),
                   balanced_error_score(),
                   balanced_accuracy_score().
        
        This class uses confusion matrix to calculate different accuracy scores
        errors for predicted values.
        
        '''
       
        self.truth = truth
        self.pred = pred
        self.createConfusionMatrix()
        
    def createConfusionMatrix(self):
        
        '''
        
        input: (NULL)
        output: (NULL)
        
        This function creates a confussion matrix from the true values
        and predicted values.
        
        '''
        
        if len(self.truth) != len(self.pred):
            raise ValueError('The input lists must be of equal lengths')
        
        TP, TN = 0, 0
        FP, FN = 0, 0
        
        for (Y, predY) in zip(self.truth, self.pred):
            if Y == 0:
                if predY == 0:
                    TP += 1
                
                elif predY == 1:
                    FN += 1
            
            elif Y == 1:
                if predY == 0:
                    FP += 1
                
                elif predY == 1:
                    TN += 1
        
        pos = [TP, FN]
        neg = [FP, TN]
        
        self.confusionMatrix = [pos, neg]
        
    def accuracy_score(self):
       
        '''
        
        input: (NULL)
        output: (float)
        
        This function uses the confusion matrix to caculate the accuracy score
        for the given prediction values.
        
        '''
        
        TP = self.confusionMatrix[0][0]
        TN = self.confusionMatrix[1][1]
        
        P = sum(self.confusionMatrix[0])
        N = sum(self.confusionMatrix[1])
        
        num = TP + TN
        din = P + N
        
        if din == 0:
            raise ValueError('Please input the predictions in 0/1')
        return num/din
    
    def balanced_error_score(self):
        
        '''
        
        input : (NULL)
        output: (float)
        
        This function uses the confusion matrix to calculate the balanced 
        error score for the predicted values.
        
        '''
        
        TP, FN = self.confusionMatrix[0]
        FP, TN = self.confusionMatrix[1]
        
        term1 = FN/(TP + FN)
        term2 = FP/(FP + TN)
        
        res = (term1 + term2)/2
        
        return res
    
    def balanced_accuracy_score(self):
        
        '''
        
        input : (NULL)
        output: (float)
        
        This function uses the confusion matrix to calculate the balanced 
        accuracy score for the predicted values.
        
        '''
        
        TP, FN = self.confusionMatrix[0]
        FP, TN = self.confusionMatrix[1]
        
        term1 = TP/(TP + FN)
        term2 = TN/(FP + TN)
        
        res = (term1 + term2)/2
        
        return res