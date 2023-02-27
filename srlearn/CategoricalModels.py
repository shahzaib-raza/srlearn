import numpy as np


class ParsingTree:

    def __init__(self):
        self.x_matrix = None
        self.y_matrix = None

    def fit(self, x_train, y_train):
        self.x_matrix = x_train
        self.y_matrix = y_train

    def parse_tree(self, x_test):
        
        # Make a copy of original training matrix to perform filtering
        data = self.x_matrix
        y_ = self.y_matrix
        
        # Iterator for iterating over features in test array
        j = 0

        # Iterator for columns in feature matrix
        for k in range(len(self.x_matrix[0, :])):

            # Selected column
            col = data[:, k]

            # Iterating over the features inside the selected column
            for i in set(col):

                # If the selected feature of the column is the feature of test array
                if x_test[j] == i:

                    # Indices of both x and y matrix of model where feature is the selected feature in that column
                    ind = [idx for idx, val in enumerate(col) if val == i]
                    
                    # Filtering data based on indices
                    data = data[ind, :]
                    y_ = y_[ind]
                    
                    # If we have a single output we break the loop as we reach our leaf node already
                    if len(set(list(y_))) == 1:
                        return y_[0]
            
            j += 1
        
        # If there is no unique output it must return the combination of outputs that may give a decision
        return list(set(y_))


    def predict(self, x_test):

        # Base case if we have only 1 scenario to predict
        if np.array(x_test).ndim == 1:
            return [self.parse_tree(x_test)]
        
        # Else predict every scenario and return list of predictions
        y_ = []
        for x_ in x_test:
            y_.append(self.parse_tree(x_))
        return y_