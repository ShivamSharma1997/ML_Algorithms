class NaiveBayes:
    
    '''
    input: (NULL)
    
    functions: ones(dim),
               zeros(dim),
               argmin(vals),
               train(trainX, trainY), 
               prediction(testX) 
    
    This class predicts values using Naive Bayes Algorithm.
    
    '''
    
    def __init__(self):
        pass
    
    def ones(self, dim):
        
        '''
        
        input: (list)
        output: (list)
        
        This function takes list of dimensions as input and returns
        a list of 0.01 with the given dimensions.
        
        '''
        
        if len(dim) not in [1,2]:
            raise ValueError('Please input 2D dimensions only.')
        
        if len(dim) == 2:
            rows, colms = dim
        
        elif len(dim) == 1:
            rows, colms = 1, dim[0]
        
        res = []
        
        for _ in range(rows):
            temp = []
            for _ in range(colms):
                temp.append(0.01)
            res.append(temp)
        
        if len(dim) == 2:
            return res
        
        elif len(dim) == 1:
            return res[0]
    
    def zeros(self, dim):
        
        '''
        
        input: (list)
        output: (list)
        
        This function takes list of dimensions as input and returns
        a list of zeroes with the given dimensions.
        
        '''
        
        if len(dim) not in [1,2]:
            raise ValueError('Please input 2D dimensions only.')
        
        if len(dim) == 2:
            rows, colms = dim
        
        elif len(dim) == 1:
            rows, colms = 1, dim[0]
        
        res = []
        
        for _ in range(rows):
            temp = []
            for _ in range(colms):
                temp.append(0.0)
            res.append(temp)
        
        if len(dim) == 2:
            return res
        
        elif len(dim) == 1:
            return res[0]
    
    def argmin(self, vals):
        
        '''
        
        input : (list)
        output: (value)
        
        This function returns the index of the minimum in a list of values.
        
        '''
        
        if type(vals[0]) != int and type(vals[0]) != float:
            raise ValueError('Please input a list of integers or floats')
        
        listToIndex = {}
        
        for i, val in enumerate(vals):
            listToIndex[val] = i
        
        minVal = sorted(vals)[0]
        minIndex = listToIndex[minVal]
        
        return minIndex
    
    def train(self, trainX, trainY):
        
        '''
        
        input: (list, list)
        output: (NULL)
        
        This function takes two lists, data and labels, respectively, and
        calculates Means and Variances to be used for predictions.
        The Means and Variances are shared within class.
        
        '''
        
        self.M = self.ones([len(set(trainY)), len(trainX[0])])
        self.var = self.ones([len(set(trainY)), len(trainX[0])])
        self.count = self.zeros([len(set(trainY))])

        LabeltoIndex = {}

        for i, label in enumerate(set(trainY)):
            LabeltoIndex[label] = i
        
        for label in trainY:
            idx = LabeltoIndex[label]
            self.count[idx] += 1
        
        for i, X in enumerate(trainX):
            for j, x in enumerate(X):
                label = trainY[i]
                idx = LabeltoIndex[label]
                self.M[idx][j] += x
        
        for i, Mlist in enumerate(self.M):
            for j, mean in enumerate(Mlist):
                self.M[i][j] = mean/self.count[i]
        
        for i, X in enumerate(trainX):
            for j, x in enumerate(X):
                label = trainY[i]
                idx = LabeltoIndex[label]
                self.var[idx][j] += (x - self.M[idx][j])**2

        for i, Varlist in enumerate(self.var):
            for j, variance in enumerate(Varlist):
                self.var[i][j] = variance/self.count[i]
            
    def prediction(self, testX):
        
        '''
        
        input: (list)
        output: (list)
        
        This function uses Means and Variances caluclated in train function
        to predict labels for the input list of test data.
        
        '''
        
        self.predict = self.zeros([len(testX), len(self.count)])
        
        for i, X in enumerate(testX):
            for k, x in enumerate(X):
                for j in range(len(self.count)):    
                    self.predict[i][j] += (x - self.M[j][k])**2/self.var[j][k]
        
        result = []
        
        for pred in self.predict:
            result.append(self.argmin(pred))
        
        return result        