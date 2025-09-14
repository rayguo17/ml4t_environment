
import numpy as np

class BagLearner(object):
    def __init__(self, learner:object,kwargs:dict, bags:int,boost=False, verbose = False):
        self.learners = []
        for i in range(bags):
            self.learners.append(learner(**kwargs))
        self.bags = bags
        pass
    
    def author(self):
        return "taung30"
    
    def add_evidence(self, data_x,data_y):
        for i in range(self.bags):
            train_ind = np.random.choice(data_x.shape[0],data_x.shape[0],replace=True)
            xtrain = data_x[train_ind]
            ytrain = data_y[train_ind]
            self.learners[i].add_evidence(xtrain,ytrain)
    
    def query(self, points:np.ndarray):
        res = np.zeros(points.shape[0])
        for i in range(self.bags):
            res = res  + self.learners[i].query(points)
        res = res / self.bags
        return res
            
        
            