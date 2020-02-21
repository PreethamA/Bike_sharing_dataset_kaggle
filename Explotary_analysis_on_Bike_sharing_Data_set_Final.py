import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import pickle
import os.path
from sklearn.metrics import mean_absolute_error
from sklearn import metrics 
from sklearn.model_selection import train_test_split

class RandomForest():



    def __init__(self,data_link='/home/preetham/Downloads/Bike-Sharing-Dataset/hour.csv'):
        try:
            self.data=pd.read_csv(data_link)
        except:
            print('Data was not found')    
        
        #print(self.data)
        # Extracting features and label of data
        self.features=self.data.drop(['dteday', 'casual','yr', 'registered','instant','cnt'],axis=1)
        self.labels=self.data['cnt']

        # Random forest model 
        self.model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='sqrt', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=2, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=1800, n_jobs=None,
           oob_score=False, random_state=100, verbose=0, warm_start=False)    

    def train(self):
        '''Train the random forest model on the training data.
        
        Keyword Arguments:
            features {pandas data frame} -- Training samples
            labels {list of int} -- Training labels
        '''
        

        assert(len(self.features) == len(self.labels))
        X_train, X_test, y_train, y_test= train_test_split(self.features,
                                                        self.labels,test_size=0.3)
        print("Spliting the data in 70-30 ratio for training and testing")                                                
        self.model.fit(X_train, y_train)

        print("Mean Absolute Error of test data_set:", mean_absolute_error(self.model.predict(X_test),y_test))
        
    def _saveModel(self, model_file='/home/preetham/model.pth'):
        ''' Store the random forest regressor model on the disk.
        Arguments:
            model_file {str} -- Model path (default: {'model.pth'})        
        Returns:
            boolean -- success of data storage
        '''

        success = False
        if self.model is not None:
            pickle.dump(self.model, open(model_file, 'wb'))
            success = True
        return success

    def randomizedParameterSearch(self, iter=100):
        ''' Defines a parameter grid and performs a random search using three fold cross validation 
        to estimate the best parameter set for the random forest data model.
        
        Keyword Arguments:
            iter {int} -- Number of search iterations. (default: {100})        
        Returns:
            [dict of str] -- Dictionary with the best random forest parameters found in this search.
        '''


        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 10, stop = 1000, num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf}

        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        rf = RandomForestRegressor()
        # Random search of parameters, using 3 fold cross validation, 
        # search across 100 different combinations, and use all available cores
        self.model = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter =iter, cv = 3, verbose=2, random_state=0, n_jobs=-1)
        self.model.fit(self.features,self.labels)    
        print("Best parameter for random forest regressor:",self.model.best_params_)
        return self.model.best_params_



def main():
        
    
    
    # Intialize to read data from CSV file and random forest model
    model = RandomForest()
    # searches the best fit random forest parameters
    model.randomizedParameterSearch()
    # trains the random forest using best fit parameters 
    model.train()
    # saving the model       
    #model._saveModel()    
    return 0
if __name__ == "__main__":
   main()  