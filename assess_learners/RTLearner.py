import numpy as np

class TreeNode:
    def __init__(self, feature=None, value=None, left=None, right=None, is_leaf=False, prediction=None):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.prediction = prediction

class RTLearner(object):
    """
    This is a Decision Tree Learner. It is implemented correctly.

    :param leaf_size: The minimum number of samples to be at a leaf node. If a node has
        less than or equal to leaf_size samples, it will become a leaf node.
        Default is 1.
    :type leaf_size: int
    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """

    def __init__(self, leaf_size=1, verbose=False):
        """
        Constructor method
        """
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "taung30" 

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner

        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        self.tree = self.build_tree(data_x, data_y)

    def build_tree(self, data_x, data_y):
        if data_x.shape[0] <= self.leaf_size: # smaller than leaf size
            return TreeNode(None, None, None, None, True, data_y.mean())
        if np.all(data_y == data_y[0]):
            return TreeNode(None, None, None, None, True, data_y[0])
        
        random_feature = np.random.randint(0, data_x.shape[1])
        split_val = np.median(data_x[:,random_feature])
        left_indices = data_x[:,random_feature] <= split_val
        right_indices = data_x[:,random_feature] > split_val
        
        left_tree = self.build_tree(data_x[left_indices],data_y[left_indices])
        right_tree = self.build_tree(data_x[right_indices],data_y[right_indices])
        
        return TreeNode(random_feature,split_val,left_tree,right_tree,False,None)
    
    def query(self, points):
        """
        Estimate a set of test points given the model we built.
        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        if self.tree is None:
            raise ValueError("The model has not been trained yet. Please call add_evidence() before query().")

        predictions = np.array([self.query_point_for_subtree(point, self.tree) for point in points])
        return predictions
         
    def query_point_for_subtree(self, point, node:TreeNode): # return prediction.
        if node.is_leaf:
            return node.prediction
        if point[node.feature] <= node.value:
            return self.query_point_for_subtree(point,node.left)
        else:
            return self.query_point_for_subtree(point,node.right)
    def print_tree(self):
        if self.tree is None:
            raise ValueError("The model has not been trained yet.")
        self.print_subtree(self.tree)
        
    def print_subtree(self, tree:TreeNode):
        print(f"IsLeaf: {tree.is_leaf}, Feature: {tree.feature}, splitVal: {tree.value}, prediction: {tree.prediction}")
        if tree.is_leaf:
            return
        self.print_subtree(tree.left)
        self.print_subtree(tree.right)
                
if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")