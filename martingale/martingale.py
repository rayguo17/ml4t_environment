""""""  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""Assess a betting strategy.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
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
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Student Name: Tin Tun Aung (replace with your name)  		  	   		 	 	 		  		  		    	 		 		   		 		  
GT User ID: taung30 (replace with your User ID)  		  	   		 	 	 		  		  		    	 		 		   		 		  
GT ID: 904042713 (replace with your GT ID)  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import numpy as np
import matplotlib.pyplot as plt  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
def author():  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    return "taung30"  # replace tb34 with your Georgia Tech username.  		  	   		 	 	 		  		  		    	 		 		   		 		  

def study_group():
    return "taung30"   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
def gtid():  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :return: The GT ID of the student  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :rtype: int  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    return 904042713  # replace with your GT ID number  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
def get_spin_result(win_prob):  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param win_prob: The probability of winning  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type win_prob: float  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :return: The result of the spin.  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :rtype: bool  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    result = False  		  	   		 	 	 		  		  		    	 		 		   		 		  
    if np.random.random() <= win_prob:  		  	   		 	 	 		  		  		    	 		 		   		 		  
        result = True  		  	   		 	 	 		  		  		    	 		 		   		 		  
    return result  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
def test_code():  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    Method to test your code  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    win_prob = 9.0/19.0  # set appropriately to the probability of a win  		  	   		 	 	 		  		  		    	 		 		   		 		  
    np.random.seed(gtid())  # do this only once  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(get_spin_result(win_prob))  # test the roulette spin  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # add your code here to implement the experiments
    figure_1(win_prob)
    plt.clf()
    figure_2(win_prob)
    plt.clf()
    figure_3(win_prob)
    plt.clf()
    figure_4(win_prob)
    plt.clf()
    figure_5(win_prob)
    plt.clf()

def playground(win_prob):
    num_episodes = 1000
    episodes_res = run_episodes(win_prob,num_episodes,bank_roll=256)
    # np.savetxt("foo.csv", episodes_res, delimiter=",")
    last_values = episodes_res[:,-1]
    unique_values, counts = np.unique(last_values,return_counts=True)
    ratios = counts/len(last_values)
    # Print results
    for value, count, ratio in zip(unique_values, counts, ratios):
        print(f"Value: {value}, Count: {count}, Ratio: {ratio:.4f}")
    #print(episodes_res)
    
    
    
    # num_true = np.sum(meet_80)
    # print(num_true)

def run_episodes(win_prob, num_episodes,bank_roll=1000000000) ->np.ndarray:
    
    episodes_res = np.zeros([num_episodes,1001])
    for i in range(0, num_episodes):
        run_episode(win_prob,i,episodes_res,bank_roll)
    return episodes_res

def figure_2(win_prob):
    num_episodes=1000
    episodes_res = run_episodes(win_prob,num_episodes)
    
    mean_col = episodes_res.mean(axis=0)
    std_col = episodes_res.std(axis=0)
    mean_plus_std = mean_col + std_col
    mean_minus_std = mean_col - std_col
    plt.xlabel("Spin round")
    plt.ylabel("Winnings")
    plt.title("Mean Winnings with Standard Deviation Across 1000 Episodes")
    plt.xlim((0,300))
    plt.ylim((-256,100))
    # plt.plot(std_col, label="standard deviation")
    plt.plot(mean_col,label="mean")
    plt.plot(mean_plus_std,label="mean with positive standard deviation")
    plt.plot(mean_minus_std, label="mean with negative standard deviation")
    plt.legend()
    # plt.show()
    plt.savefig(fname="./images/figure2.png", format="png")
    print(mean_col,std_col)

def figure_3(win_prob):
    num_episodes=1000
    np_episodes = run_episodes(win_prob,num_episodes)
    median_col = np.median(np_episodes,axis=0)
    std_col = np_episodes.std(axis=0)
    median_plus_std = median_col + std_col
    median_minus_std = median_col - std_col
    plt.xlabel("Spin round")
    plt.ylabel("Winnings")
    plt.title("Median Winnings with Standard Deviation Across 1000 Episodes")
    plt.xlim((0,300))
    plt.ylim((-256,100))
    # plt.plot(std_col, label="standard deviation")
    plt.plot(median_col,label="median")
    plt.plot(median_plus_std,label="median with positive standard deviation")
    plt.plot(median_minus_std, label="median with negative standard deviation")
    plt.legend()
    # plt.show()
    plt.savefig(fname="./images/figure3.png", format="png")
    # print(mean_col,std_col)

def figure_1(win_prob):
    num_episodes = 10
    plt.xlabel("Spin round")
    plt.ylabel("Winnings")
    plt.xlim((0,300))
    plt.ylim((-256,100))
    episodes_res = run_episodes(win_prob,num_episodes)
    
    for i in range(0,num_episodes):
        plt.plot(episodes_res[i],label="episode {}".format(i+1))
    plt.legend()
    plt.title("Simulated Winnings Across 10 Episodes")
    # plt.show()
    plt.savefig(fname="./images/figure1.png",format="png")
    
def figure_4(win_prob):
    num_episodes=1000
    episodes_res = run_episodes(win_prob,num_episodes,bank_roll=256)
    
    mean_col = episodes_res.mean(axis=0)
    std_col = episodes_res.std(axis=0)
    mean_plus_std = mean_col + std_col
    mean_minus_std = mean_col - std_col
    plt.xlabel("Spin round")
    plt.ylabel("Winnings")
    plt.title("Mean Winnings with Standard Deviation Across 1000 Episodes")
    plt.xlim((0,300))
    plt.ylim((-256,100))
    # plt.plot(std_col, label="standard deviation")
    plt.plot(mean_col,label="mean")
    plt.plot(mean_plus_std,label="mean with positive standard deviation")
    plt.plot(mean_minus_std, label="mean with negative standard deviation")
    plt.legend()
    # plt.show()
    plt.savefig(fname="./images/figure4.png", format="png")
    print(mean_col,std_col)
    
def figure_5(win_prob):
    num_episodes=1000
    np_episodes = run_episodes(win_prob,num_episodes,bank_roll=256)
    median_col = np.median(np_episodes,axis=0)
    std_col = np_episodes.std(axis=0)
    median_plus_std = median_col + std_col
    median_minus_std = median_col - std_col
    plt.xlabel("Spin round")
    plt.ylabel("Winnings")
    plt.title("Median Winnings with Standard Deviation Across 1000 Episodes")
    plt.xlim((0,300))
    plt.ylim((-256,100))
    # plt.plot(std_col, label="standard deviation")
    plt.plot(median_col,label="median")
    plt.plot(median_plus_std,label="median with positive standard deviation")
    plt.plot(median_minus_std, label="median with negative standard deviation")
    plt.legend()
    # plt.show()
    plt.savefig(fname="./images/figure5.png", format="png")
    # print(mean_col,std_col)
 	 	 		  		  		    	 		 		   		 		  
def run_episode(win_prob, episode_index, array,bank_roll=10000000000):
    bet_cnt = 1
    episode_winnings = 0
    while bet_cnt < 1000 and episode_winnings<80 and bank_roll>0:
        bet_amount = 1
        won = False
        #num_bets_to_win = 0
        while (not won) and (bet_cnt < 1000 and bank_roll>0):
            won = get_spin_result(win_prob=win_prob)
            #num_bets_to_win += 1
            if won:
                episode_winnings += bet_amount
                bank_roll += bet_amount
            else:
                episode_winnings -= bet_amount
                bank_roll -= bet_amount
                # if bank_roll<=0:
                #     print(f"Number of bets to lose: {num_bets_to_win}, bet_amount: {bet_amount}") 
                bet_amount = min(bet_amount*2, bank_roll)
                
            array[episode_index,bet_cnt] = episode_winnings
            
            bet_cnt+=1
        #print("num of bets to win: ", num_bets_to_win)    
        
            
    while bet_cnt <=1000:
        array[episode_index,bet_cnt] = episode_winnings
        bet_cnt+=1
    bet_amount += 1
    
        
        	  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 	 	 		  		  		    	 		 		   		 		  
    test_code()  		  	   		 	 	 		  		  		    	 		 		   		 		  
