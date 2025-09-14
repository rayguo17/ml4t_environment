

import numpy as np

class TreeNode:
    def __init__(self, feature=None, value=None, left=None, right=None, is_leaf=False, prediction=None):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.prediction = prediction

class DTLearner(object):
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
    # For each ndarray representing a Node, each field is defined as below:
    # [feature, splitval, left_child_node_offset, right_child_node_offset, is_leaf, prediction]
    def build_tree(self, data_x, data_y)->np.ndarray:
        if data_x.shape[0] <= self.leaf_size: # smaller than leaf size
            return np.array([
                None, None, None, None, True, data_y.mean()
            ])
        if np.all(data_y == data_y[0]):
            return np.array([[None, None, None, None, True, data_y[0]]])
        if np.all(data_x == data_x[0]):
            return np.array([[None, None, None, None, True,data_y.mean()]])
        
        best_feature = self.determine_best_feature(data_x,data_y)
        split_val = np.median(data_x[:,best_feature])
        
        left_indices = data_x[:,best_feature] <= split_val
        right_indices = data_x[:,best_feature] > split_val
        
        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return np.array([[None, None, None, None, True, data_y.mean()]])
        
        left_tree = self.build_tree(data_x[left_indices],data_y[left_indices])
        right_tree = self.build_tree(data_x[right_indices],data_y[right_indices])
        left_tree_len = 0
        if len(left_tree.shape) == 1:
            left_tree_len = 2
        else:
            left_tree_len = left_tree.shape[0]+1
        return np.vstack(
            [np.array([[
                best_feature,split_val,1,left_tree_len,False,None
            ]]),
            left_tree,
            right_tree]
        )
    
    # Find the feature that have highest RMS reduction.
    def determine_best_feature(self, data_x, data_y):
        best_score = 0.0
        best_feature = -1
        for feature in range(data_x.shape[1]):
            score = np.corrcoef(data_x[:,feature], data_y)[0,1]
            abs_score = np.abs(score)
            if abs_score > best_score:
                best_score = abs_score
                best_feature = feature
        return best_feature
    
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

        predictions = np.array([self.query_point_for_subtree(point,offset=0) for point in points])
        return predictions
         
    def query_point_for_subtree(self, point,offset:int): # return prediction.
        node = self.tree[offset]
        if node[4]:
            return node[5]
        if point[node[0]] <= node[1]:
            return self.query_point_for_subtree(point, offset+ node[2])
        else:
            return self.query_point_for_subtree(point, offset+ node[3])
    
    def print_tree(self):
        if self.tree is None:
            raise ValueError("The model has not been trained yet.")
        self.print_subtree(offset=0)
        
    def print_subtree(self,offset:int):
        tree = self.tree[offset]
        print(f"IsLeaf: {tree[4]}, Feature: {tree[0]}, splitVal: {tree[1]}, prediction: {tree[5]}, left_child: {tree[2]}, right_child: {tree[3]}, offset: {offset}")
        if tree[4]:
            return
        self.print_subtree(tree[2] + offset) # try to print left tree.
        self.print_subtree(tree[3] + offset) # try to print right tree.
                
if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")