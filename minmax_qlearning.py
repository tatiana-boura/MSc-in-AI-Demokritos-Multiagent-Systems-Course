import numpy as np
import random
from scipy.optimize import minimize
import os
"""
we are modelling the game below:

        ######################
        #      C         D   #
        # A (3, -3)   (1, -1)#
        # B (2, -2)   (4, -4)#
        #####################
"""

num_of_actions = 2
"""
player1_payoffs = [[3, 1],
                   [2, 4]]
"""

# matching pennies
player1_payoffs = [[1, -1],
                   [-1, 1]]

Q_values = np.ones((num_of_actions, num_of_actions))
# V_values = np.ones((num_of_actions, num_of_actions))
V_value = 1

P = np.ones((num_of_actions, num_of_actions)) / 4

learning_rate = 1.0
gamma = 0.9
explor = 0.6


def action_choice():
    if random.choices([0, 1], weights=(explor * 100, (1 - explor) * 100))[0] == 0:
        return random.choice([0, 1]), random.choice([0, 1])
    else:
        return random.choices([(0, 0), (0, 1),
                               (1, 0), (1, 1)],
                              weights=(P[0][0] * 100, P[0][1] * 100, P[1][0] * 100, P[1][1] * 100))[0]


curr_episodes = 0
total_num_of_episodes = 1000
def mycon(x):
    return np.sum(x) -1

f = lambda x: min(np.sum((x*Q_values.flatten()).reshape(2,2), axis=0))
cons = ({'type': 'eq', 'fun': mycon }) #overflow check




while curr_episodes < total_num_of_episodes:
    action = action_choice()

    reward = player1_payoffs[action[0]][action[1]]

    Q_values[action[0]][action[1]] = (1 - learning_rate) * Q_values[action[0]][action[1]] + \
                                     learning_rate * (reward + gamma * V_value)

    P = minimize(fun = lambda x: -f(x), x0 = np.array([0.1, 0.2, 0.2, 0.5]), constraints=cons).x.reshape(2,2)
    print(P)
    print(np.sum(P))
    print(P.shape)
    os._exit(1)

    V_value = f(P.flatten())

    curr_episodes += 1

print("Q values:")
print(Q_values)
print("V value:")
print(V_value)
print("Policies")
print(P)
