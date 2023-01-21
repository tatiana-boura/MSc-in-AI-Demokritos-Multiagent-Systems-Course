import numpy as np
import random
from scipy.optimize import minimize

"""
we are modelling the game below:

        ######################
        #      C         D   #
        # A (3, -3)   (1, -1)#
        # B (2, -2)   (4, -4)#
        #####################
"""

player1_payoffs = [[3, 1],
                   [2, 4]]
"""

# matching pennies
player1_payoffs = [[1, -1],
                   [-1, 1]]
"""
num_of_actions = 2

# minmax Q learning settings
Q_values = np.ones((num_of_actions, num_of_actions))
V_value = 1
P = np.ones((num_of_actions, num_of_actions)) / 4

learning_rate = 1.0
gamma = 0.9
explor = 0.5
curr_episodes = 0
total_num_of_episodes = 1000


def action_choice():
    if random.choices([0, 1], weights=(explor * 100, (1 - explor) * 100))[0] == 0:
        return random.choice([0, 1]), random.choice([0, 1])
    else:
        return random.choices([(0, 0), (0, 1),
                               (1, 0), (1, 1)],
                              weights=(P[0][0] * 100, P[0][1] * 100, P[1][0] * 100, P[1][1] * 100))[0]


bnds = ((0., 1.), (0., 1.), (0., 1.), (0., 1.))
cons = ({'type': 'eq', 'fun': lambda x: 1.0 - np.sum(x)})

f = lambda x: min(np.sum((x * (Q_values.flatten())).reshape(2, 2), axis=0))

while curr_episodes < total_num_of_episodes:

    action = action_choice()
    reward = player1_payoffs[action[0]][action[1]]
    Q_values[action[0]][action[1]] = (1 - learning_rate) * Q_values[action[0]][action[1]] + \
                                     learning_rate * (reward + gamma * V_value)

    try:
        P = minimize(fun=lambda x: -f(x), x0=np.array([0., 0., 0., 0.]), constraints=cons, bounds=bnds).x.reshape(2, 2)

        if np.abs(np.sum(P)) > 1.1:
            print(np.sum(P))
            raise ValueError("Optimization failed")
    finally:

        V_value = f(P.flatten())
        curr_episodes += 1

        if curr_episodes % 100 == 0:
            explor = -0.05 + explor if explor > 0.05 else explor
            learning_rate = -0.05 + learning_rate if learning_rate > 0.05 else learning_rate

print("Q values:")
print(Q_values)
print("V value:")
print(V_value)
print("Policies")
print(P)
print(f"learning rate : {learning_rate}")
print(f"explor : {explor}")


