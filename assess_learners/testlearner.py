""""""  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		 	 	 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		 	 	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 		  		  		    	 		 		   		 		  
or edited.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		 	 	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		 	 	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import math  		  	   		 	 	 		  		  		    	 		 		   		 		  
import sys  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import numpy as np  
import matplotlib.pyplot as plt		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import LinRegLearner as lrl  
import DTLearner as dl		
import RTLearner as rl  
import BagLearner as bl
import InsaneLearner as il	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 
def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the R-squared (coefficient of determination) metric.

    :param y_true: True target values
    :type y_true: numpy.ndarray
    :param y_pred: Predicted target values
    :type y_pred: numpy.ndarray
    :return: R-squared value
    :rtype: float
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

# [(leaf_size, DTModel)] Would contain multiple DTModel with same leaf_size, to avoid the effect of randomization.
# 1. For each model, compute the in-sample and out-of-sample RMSE and correlation. Store the results in a numpy array.
# take the average of the results for each leaf_size.
# 2. Return a numpy array of shape (n, 5) where n is the number of unique leaf sizes.
# Each row contains: [leaf_size, avg_in_sample_RMSE, avg_in_sample_corr, avg_out_of_sample_RMSE, avg_out_of_sample_corr]
def evaluate_models(models: list[tuple[int,dl.DTLearner]]) -> np.ndarray :
    results = []
    for model in models:
        leaf_size, dt_learner = model
        # leaf_size
        in_sample_rmse = math.sqrt(((train_y - dt_learner.query(train_x)) ** 2).sum() / train_y.shape[0])
        
        in_sample_r2 = calculate_r2(train_y, dt_learner.query(train_x))
        in_sample_corr = np.corrcoef(dt_learner.query(train_x), y=train_y)[0, 1]
        in_sample_explained_variance = 1 - np.var(train_y - dt_learner.query(train_x)) / np.var(train_y)
        out_of_sample_rmse = math.sqrt(((test_y - dt_learner.query(test_x)) ** 2).sum() / test_y.shape[0])
        out_of_sample_corr = np.corrcoef(dt_learner.query(test_x), y=test_y)[0, 1]
        out_of_sample_r2 = calculate_r2(test_y, dt_learner.query(test_x))
        out_ofsample_explained_variance = 1 - np.var(test_y - dt_learner.query(test_x)) / np.var(test_y)

        results.append([leaf_size, in_sample_rmse, in_sample_corr, out_of_sample_rmse, out_of_sample_corr, in_sample_r2, out_of_sample_r2, in_sample_explained_variance, out_ofsample_explained_variance])
    res = np.array(results)
    unique_leaf_sizes = np.unique(res[:, 0])
    final_results = []
    for leaf_size in unique_leaf_sizes:
        leaf_size_results = res[res[:, 0] == leaf_size]
        avg_in_sample_rmse = np.mean(leaf_size_results[:, 1])
        avg_in_sample_corr = np.mean(leaf_size_results[:, 2])
        avg_in_sample_r2 = np.mean(leaf_size_results[:, 5])
        avg_in_sample_explained_variance = np.mean(leaf_size_results[:, 7])
        avg_out_of_sample_rmse = np.mean(leaf_size_results[:, 3])
        avg_out_of_sample_corr = np.mean(leaf_size_results[:, 4])
        avg_out_of_sample_r2 = np.mean(leaf_size_results[:, 6])
        avg_out_ofsample_explained_variance = np.mean(leaf_size_results[:, 8])
        final_results.append([leaf_size, avg_in_sample_rmse, avg_in_sample_corr, avg_out_of_sample_rmse, avg_out_of_sample_corr, avg_in_sample_r2, avg_out_of_sample_r2, avg_in_sample_explained_variance, avg_out_ofsample_explained_variance])
    return np.array(final_results)

def test_one_model(train_x, train_y):
    learner = dl.DTLearner(leaf_size=1)
    learner.add_evidence(train_x, train_y)  # train it 
    print(learner.author())  	
    # learner.print_tree()
    # evaluate in sample  		  	   		 	 	 		  		  		    	 		 		   		 		  
    pred_y = learner.query(train_x)  # get the predictions  		  	   		 	 	 		  		  		    	 		 		   		 		  
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print()  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print("In sample results")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"RMSE: {rmse}")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    c = np.corrcoef(pred_y, y=train_y)  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"corr: {c[0,1]}")  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # evaluate out of sample  		  	   		 	 	 		  		  		    	 		 		   		 		  
    pred_y = learner.query(test_x)  # get the predictions  		  	   		 	 	 		  		  		    	 		 		   		 		  
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print()  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print("Out of sample results")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"RMSE: {rmse}")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    c = np.corrcoef(pred_y, y=test_y)  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"corr: {c[0,1]}")  	
    
def train_dt_model(train_x, train_y):
    pass


def plot_RMSE_results(results: np.ndarray, rt_results: np.ndarray):
    plt.title("RT vs DT RMSE")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.plot(results[:,0], results[:,1], label="DT In-sample RMSE")
    plt.plot(results[:,0], results[:,3], label="DT Out-of-sample RMSE")
    plt.plot(rt_results[:,0], rt_results[:,1], label="RT In-sample RMSE", linestyle='dashed')
    plt.plot(rt_results[:,0], rt_results[:,3], label="RT Out-of-sample RMSE", linestyle='dashed')
    plt.legend()
    # plt.show()
    plt.savefig("./images/rmse_metric.png")
    pass

def plot_results(results: np.ndarray, rt_results: np.ndarray):
    plt.title("RT vs DT R2")
    plt.xlabel("Leaf Size")
    plt.ylabel("R2")
    plt.plot(results[:,0], results[:,5], label="DT In-sample R2")
    plt.plot(results[:,0], results[:,6], label="DT Out-of-sample R2")
    plt.plot(rt_results[:,0], rt_results[:,5], label="RT In-sample R2", linestyle='dashed')
    plt.plot(rt_results[:,0], rt_results[:,6], label="RT Out-of-sample R2", linestyle='dashed')
    plt.legend()
    # plt.show()
    plt.savefig("./images/r2_metric.png")
    pass

def plot_explained_variance(results: np.ndarray, rt_results: np.ndarray):
    plt.title("RT vs DT Explained Variance")
    plt.xlabel("Leaf Size")
    plt.ylabel("Explained Variance")
    plt.plot(results[:,0], results[:,7], label="DT In-sample Explained Variance")
    plt.plot(results[:,0], results[:,8], label="DT Out-of-sample Explained Variance")
    plt.plot(rt_results[:,0], rt_results[:,7], label="RT In-sample Explained Variance", linestyle='dashed')
    plt.plot(rt_results[:,0], rt_results[:,8], label="RT Out-of-sample Explained Variance", linestyle='dashed')
    plt.legend()
    # plt.show()
    plt.savefig("./images/explained_variance_metric.png")
    pass
	    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 	 	 		  		  		    	 		 		   		 		  
    if len(sys.argv) != 2:  		  	   		 	 	 		  		  		    	 		 		   		 		  
        print("Usage: python testlearner.py <filename>")  		  	   		 	 	 		  		  		    	 		 		   		 		  
        sys.exit(1)  		  	   		 	 	 		  		  		    	 		 		   		 		  
    inf = open(sys.argv[1])
    # first column is date, treat it as index.
    # first row is header, skip it.
    inf.readline()		  	   		 	 	 		  		  		    	 		 		   		 		  
    data = np.array(  		  	   		 	 	 		  		  		    	 		 		   		 		  
        [list(map(float, s.strip().split(",")[1:])) for s in inf.readlines()]  		  	   		 	 	 		  		  		    	 		 		   		 		  
    )  		  	   		 	 	 		  		  		    	 		 		   		 		  
    
    # randomly select data
    np.random.seed(42)
    permutation = np.random.permutation(data.shape[0])
    data = data[permutation]  	 		 		   		 		  
    # compute how much of the data is training and testing  		  	   		 	 	 		  		  		    	 		 		   		 		  
    train_rows = int(0.6 * data.shape[0])  		  	   		 	 	 		  		  		    	 		 		   		 		  
    test_rows = data.shape[0] - train_rows  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # separate out training and testing data  		  	   		 	 	 		  		  		    	 		 		   		 		  
    train_x = data[:train_rows, 0:-1]  		  	   		 	 	 		  		  		    	 		 		   		 		  
    train_y = data[:train_rows, -1]  		  	   		 	 	 		  		  		    	 		 		   		 		  
    test_x = data[train_rows:, 0:-1]  		  	   		 	 	 		  		  		    	 		 		   		 		  
    test_y = data[train_rows:, -1]  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # create a learner and train it  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # learner = il.InsaneLeaner()
    
    #learner = bl.BagLearner(learner=dl.DTLearner,bags=20,kwargs={"leaf_size":3,"verbose":True})  # create a LinRegLearner  		  	   		 	 	 		  		  		    	 		 		   		 		  
   		  	   		 	 	 		  		  		    	 		 		   		 		  
    dt_learners = []
    for leaf_size in range(1,50,1):
        dt_learner = dl.DTLearner(leaf_size=leaf_size, verbose=False)
        dt_learner.add_evidence(train_x, train_y)  # train it
        dt_learners.append((leaf_size, dt_learner))
        # bag_learner = bl.BagLearner(learner=dl.DTLearner,bags=20,kwargs={"leaf_size":leaf_size})  # create a LinRegLearner
        # bag_learner.add_evidence(train_x, train_y)  # train it
        # learners.append((leaf_size, bag_learner))
    results = evaluate_models(dt_learners)
    
    rt_learners = []
    for leaf_size in range(1,50,1):
        rt_learner = rl.RTLearner(leaf_size=leaf_size, verbose=False)
        rt_learner.add_evidence(train_x, train_y)  # train it
        rt_learners.append((leaf_size, rt_learner))
    rt_results = evaluate_models(rt_learners)
    
    
    plot_RMSE_results(results, rt_results)
    plt.clf()
    plot_results(results, rt_results)
    plt.clf()
    plot_explained_variance(results, rt_results)
    # plt.savefig("./images/r2_metric.png")
    
    	  	   		 	 	 		  		  		    	 		 		   		 		  

