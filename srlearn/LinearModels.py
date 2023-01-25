import random
try:
    import numpy as np
except:
    raise ImportError("cannot import numpy,  install it by using \'pip install numpy\' in command line.")

# Linear Regression Model
class LinearRegression:
    
    
    # Initializing the class
    def __init__(self):
        self.x_train: np.ndarray = np.ndarray([])
        self.y_train: np.array = np.array([])
        self.weights: np.ndarray = np.ndarray([])
        self.e: float = float()
    
    
    # Gradient descent function to optimize a weight
    def get_optimized_weights(self, x, y):
        try:
            w = np.linalg.inv(x.T @ x) @ x.T @ y
            return w
        except:
            raise Exception("Unable to compute weights for this data!")
            
    
    def get_e(self, x, y, w):
        er = []
        for i in range(len(x)):
            er.append(y[i]-(sum(x[i]*w)))
        return sum(er)/len(er)
    
    
    # Fitting the data in linear regression model
    def fit(self, x_train, y_train):
        
        # Initializing x_train data
        if type(x_train) != np.ndarray:
            try:
                x_train = np.array(x_train)
            except:
                raise TypeError("Can not convert the given data into numpy array")
                
        self.x_train = x_train
        
        # Initializing y_train data
        if type(y_train) != np.ndarray:
            try:
                y_train = np.array(y_train)
            except:
                raise TypeError("Can not convert the given data into numpy array")
        
        self.y_train = y_train
        
        # Optimized weights
        self.weights = self.get_optimized_weights(self.x_train, self.y_train)
        
        self.e = self.get_e(self.x_train, self.y_train, self.weights)
        
    
    # Function to predict
    def predict(self, x_test):
        
        # Function will predict the input vector array
        """
        x_test: numpy.ndarray
        return: numpy.array
        """
        
        if type(x_test) != np.ndarray:
            try:
                x_test = np.array(x_test)
            except:
                raise TypeError("Can not convert the given data into numpy array")
        
        return np.array([round(sum(i*self.weights), 2)+self.e for i in x_test])