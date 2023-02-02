import numpy as np
import random
from scipy.optimize import minimize
from plots import *

"""
we are modelling the game below:

        ###################
        # (3, -3)  (1, -1)#
        # (2, -2)  (4, -4)#
        ###################
"""

payoffs = [[3, 1],
           [2, 4]]

class QAgent():
    def __init__(self, explor, learning_rate, gamma, player_id):
        self.explor = explor
        self.learning_rate = learning_rate
        self.P = np.array([0.5, 0.5])
        self.Q = np.ones((2, 2))
        self.V = 1.0
        self.gamma = gamma
        self.payoffs = np.array(payoffs) * -1 if player_id == 2 else np.array(payoffs)
        self.player_id = player_id

    def take_action(self):
        if random.choices([0, 1], weights=(self.explor * 100, (1 - self.explor) * 100))[0] == 0:
            return random.choice([0, 1])
        else:
            return random.choices([0, 1], weights=(self.P[0] * 100, self.P[1] * 100))[0]

    def observe(self, action, opponent):
        return self.payoffs[action][opponent] if self.player_id == 1 else self.payoffs[opponent][action]

    def update_Q(self, action, opponent, reward):

        if self.player_id == 2:
            temp = action
            action = opponent
            opponent = temp

        self.Q[action][opponent] = (1 - self.learning_rate) * self.Q[action][opponent] + \
                                   self.learning_rate * (reward + self.gamma * self.V)

    def update_P(self, opponent):
        bnds = ((0., 1.), (0., 1.))
        cons = ({'type': 'eq', 'fun': lambda x: 1.0 - np.sum(x)})


        if self.player_id == 1:
            f = lambda  x: min(np.matmul(x.T,self.Q))
        else:
            f = lambda  x: min(np.matmul(x.T,self.Q.T))

        self.P = minimize(fun=lambda x: -f(x), x0=np.array([0., 0.]), constraints=cons, bounds=bnds).x

    def update_V(self):

        if self.player_id == 1:
            f = lambda  x: min(np.matmul(x.T,self.Q))
        else:
            f = lambda  x: min(np.matmul(x.T,self.Q.T))

        self.V = f(self.P)

def final_expected_payoff(agent1,agent2):
    expected_util_action_11 = agent1.payoffs[0][0] * agent2.P[0] + agent1.payoffs[0][1] * agent2.P[1]
    expected_util_action_12 = agent1.payoffs[1][0] * agent2.P[0] + agent1.payoffs[1][1] * agent2.P[1]

    expected_util_action_21 = agent2.payoffs[0][0] * agent1.P[0] + agent2.payoffs[1][0] * agent1.P[1]
    expected_util_action_22 = agent2.payoffs[0][1] * agent1.P[0] + agent2.payoffs[1][1] * agent1.P[1]

    return expected_util_action_11*agent1.P[0] + expected_util_action_12*agent1.P[1], expected_util_action_21*agent2.P[0] + expected_util_action_22*agent2.P[1]


curr_episode = 0
total_num_of_episodes = 1000

agent1 = QAgent(explor=0.3, learning_rate=1.0, gamma=0.9, player_id=1)
agent2 = QAgent(explor=0.3, learning_rate=1.0, gamma=0.9, player_id=2)

policies = [[agent1.P,agent2.P]]


while curr_episode < total_num_of_episodes:
    action1 = agent1.take_action()
    action2 = agent2.take_action()

    rew1 = agent1.observe(action=action1, opponent=action2)
    rew2 = agent2.observe(action=action2, opponent=action1)

    agent1.update_Q(action=action1, opponent=action2, reward=rew1)
    agent1.update_P(opponent=action2)
    agent1.update_V()

    agent2.update_Q(action=action2, opponent=action1, reward=rew2)
    agent2.update_P(opponent=action1)
    agent2.update_V()

    policies.append([agent1.P,agent2.P])

    if curr_episode % 100 == 0:
        agent1.learning_rate *= 0.8
        agent2.learning_rate *= 0.8

    curr_episode += 1


print("Agent's 1 Policy:")
print(agent1.P)

print("Agent's 2 Policy:")
print(agent2.P)

print("Agents' Expected Payoff:")
print(final_expected_payoff(agent1,agent2))

print("V:")
print(agent1.V*0.1, agent2.V*0.1)
print(agent2.V*0.1 + agent1.V*0.1)


policies = policies[:1000]
total_num_of_episodes = 1000
policy_iter(policies,total_num_of_episodes)

