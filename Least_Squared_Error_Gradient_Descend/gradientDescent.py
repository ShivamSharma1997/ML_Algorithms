from math import sqrt
from random import uniform

class gradientDescent:
    
    '''
    
    functions:  zeros(dim),
                train(trainX, trainY),
                predict(testX)
    
    This class implements the gradient descent algorithm using the square loss
    funtion to the given datasets and returns prediction on a test dataset.
    
    '''
    
    def __init__(self, eta=0.01, theta=0.01):
        
        self.eta = eta
        self.theta = theta
    
    def zeros(self, dim):
        
        '''
        
        input: (list)
        output: (list)
        
        This function takes list of dimensions as input and returns
        a list of zeroes with the given dimensions.
        
        '''
        
        if len(dim) not in [1,2]:
            raise ValueError('Please input list with 1D or 2D dimensions only.')
        
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
    
    def dotProduct(self, listA, listB):
        
        '''
        
        input: (list, list)
        output: (float)
        
        This function outputs the dot products of two given lists.
        
        '''
        
        if len(listA) != len(listB):
            raise ValueError('Please input lists with same dimensions')
        
        dot = 0
        
        for a, b in zip(listA, listB):
            dot += a*b
        
        return dot

    def squareLoss(self, data, labels, weights):
        
        '''
        
        input : (list, list, list)
        output : (float)
        
        This function calculates the objective function.
        
        '''
        
        obj = 0
        
        for line, lab in zip(data, labels):
            temp = lab - self.dotProduct(line, weights)
            obj += temp ** 2
        
        return obj
        
    def train(self, trainX, trainY):
        
        '''
        
        input: (list, list)
        output: (None)
        
        This function trains the gradient descent model on the given training 
        data and minimizes the sqaure error loss function.
        
        '''
        
        flag = 1
        wdim = len(trainX[0])+1
        self.weights, self.obj = [], 0
        
        newTrainX = []
        
        for X in trainX:
            temp = [1.0]
            temp.extend(X)
            newTrainX.append(temp)
        
        for _ in range(wdim):
            self.weights.append(uniform(-0.01, 0.01))
        
        self.obj = self.squareLoss(newTrainX, trainY, self.weights)
        
        while(flag):    
            
            delF = self.zeros([wdim])
            
            for i, X in enumerate(newTrainX):
                temp = trainY[i] - self.dotProduct(X, self.weights)
                for j, x in enumerate(X):
                    delF[j] += temp * x
            
            for i, dF in enumerate(delF):    
                self.weights[i] += self.eta*dF            
            
            newObj = self.squareLoss(newTrainX, trainY, self.weights)

            if self.obj - newObj < self.theta:
                flag = 0        

            self.obj = newObj

    def predict(self, testX):
        
        '''
        
        input: (list)
        output: (list)
        
        This function uses pre-trained weights and returns a list of prediction
        for the given test data.
        
        '''
        
        pred = []
        
        newTestX = []
        
        for X in testX:
            temp = [1.0]
            temp.extend(X)
            newTestX.append(temp)
        
        for X in newTestX:
            pred.append(self.dotProduct(X, self.weights))
        
        res = []
        
        for p in pred:
            if p > 0:
                res.append(1)
            else:
                res.append(0)
        
        return res
    
    def distToOrigin(self):
        
        '''
        
        input : (NULL)
        output : (float)
        
        This function returns the distance of the plane to the origin.
        
        '''
        
        norm = sqrt(sum([x**2 for x in self.weights[1:]]))
        w0 = self.weights[0]
        
        dist = w0/norm
        
        if dist > 0:
            return dist
        else:
            return dist*(-1)