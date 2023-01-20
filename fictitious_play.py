import numpy as np
import random

"""
we are modelling the game below:

        ###################
        # (3, -3)  (1, -1)#
        # (2, -2)  (4, -4)#
        ###################
"""


"""
player1_payoffs = [[3, 1],
                   [2, 4]]

player2_payoffs = [[-3, -1],
                   [-2, -4]]
"""


player1_payoffs = [[2, 2],
                   [1, 3]]

player2_payoffs = [[-2, -2],
                   [-1, -3]]


"""
player1_payoffs = [[1, -1],
                   [-1, -1]]

player2_payoffs = [[-1, 1],
                   [1, 1]]
"""

def best_response(player, belief):
    if player == 1:
        expected_util_action_1 = player1_payoffs[0][0] * belief[0] + player1_payoffs[0][1] * belief[1]
        expected_util_action_2 = player1_payoffs[1][0] * belief[0] + player1_payoffs[1][1] * belief[1]
    else:
        expected_util_action_1 = player2_payoffs[0][0] * belief[0] + player2_payoffs[1][0] * belief[1]
        expected_util_action_2 = player2_payoffs[0][1] * belief[0] + player2_payoffs[1][1] * belief[1]


    if expected_util_action_1 == expected_util_action_2:
        print("it's a tie")
        action = random.choice([0,1])
    else:
        action = np.argmax([expected_util_action_1, expected_util_action_2])

    return action


def observe_and_update(action, player_counts):
    if action == 0:
        player_counts[0] += 1
    else:
        player_counts[1] += 1

    total = player_counts[0] + player_counts[1]
    return (player_counts[0] / total, player_counts[1] / total), player_counts


max_iter = 20
iteration = 0
converged = False

player1_counts = [1, 1]
player2_counts = [0, 2]

player1_belief = (player1_counts[0] / (player1_counts[0] + player1_counts[1]),
                  player1_counts[1] / (player1_counts[0] + player1_counts[1]))
player2_belief = (player2_counts[0] / (player2_counts[0] + player2_counts[1]),
                  player2_counts[1] / (player2_counts[0] + player2_counts[1]))

print("Initial Beliefs")
print(player1_belief)
print(player2_belief)

while iteration < max_iter:
    action1 = best_response(1, player1_belief)
    action2 = best_response(2, player2_belief)

    player1_belief, player1_counts = observe_and_update(action2, player1_counts)
    player2_belief, player2_counts = observe_and_update(action1, player2_counts)

    iteration += 1

print('Nash Equilibria')
print(player1_belief)
print(player2_belief)
