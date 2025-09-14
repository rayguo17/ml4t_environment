import BagLearner as bl
import LinRegLearner as lrl
import numpy as np
class InsaneLeaner(object):
    def __init__(self,verbose=False):
        self.learners = [bl.BagLearner(learner=lrl.LinRegLearner, kwargs={"verbose":verbose},bags=20,verbose=verbose)]
    def add_evidence(self, data_x, data_y):
        for learner in self.learners:
            learner.add_evidence(data_x,data_y)
    def author(self):
        return "taung30"
    def query(self, points):
        res = np.zeros(points.shape[0])
        for learner in self.learners:
            res = res + learner.query(points)
        return res / len(self.learners)
                