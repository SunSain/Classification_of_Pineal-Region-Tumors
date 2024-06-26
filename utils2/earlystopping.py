from turtle import Turtle
import numpy as np

class EarlyStopping:

    def __init__(self, patience=80, verbose = False):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf


    def __call__(self, val_metric):
        
        if self.best_score is None:
            self.best_score = val_metric
<<<<<<< HEAD
        elif self.best_score < val_metric:
=======
        elif self.best_score > val_metric: # have no growth
>>>>>>> 3a4a4f2 (20240625-code)
            self.counter +=1
            print("EarlyStopping counter : {} out of {}\n".format(self.counter,self.patience))
            if self.counter > self.patience:
                self.early_stop = True    
        else:
            self.best_score = val_metric
            self.counter = 0