import numpy as np
import random

"""
we are modelling the game below:

        ###################
        # (3, -3)  (1, -1)#
        # (2, -2)  (4, -4)#
        ###################
"""

num_of_actions = 2
#num_of_states = 1

player1_payoffs = [[3, 1],
                   [2, 4]]

player2_payoffs = [[-3, -1],
                   [-2, -4]]

Q_values = np.ones((num_of_actions,num_of_actions))
V_values = np.ones((num_of_actions,num_of_actions))

P = np.ones((num_of_actions,num_of_actions)) / 2


learning_rate = 1.0
beta = 0.9
explor = 0.6

def action_choice(player):
    if random.choices([0, 1],weights=(explor*100, (1-explor)*100))[0] == 0 :
        return random.choice([0, 1])
    else:
        return random.choices([0, 1], weights=(P[player-1][0] * 100, P[player-1][1] * 100))[0]


curr_episodes = 0
total_num_of_episodes = 2
while curr_episodes < total_num_of_episodes:

    action1 = action_choice(1)
    action2 = action_choice(2)

    reward1 = player1_payoffs[action1][action2]

    Q_values[action1][action2] = (1-learning_rate) * Q_values[action1][action2] + learning_rate * (reward1 + beta * V_values[action1][action2])

    #print(action1, action2)
    #print(reward1,reward2)

    curr_episodes += 1