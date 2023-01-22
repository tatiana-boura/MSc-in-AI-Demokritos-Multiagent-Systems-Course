import matplotlib.pyplot as plt
plt.style.use('bmh')
#import numpy as np


def policy_iter(policy_list, iter=1000):

    # Get policies of each agent
    agent1_policies = [policy[0] for policy in policy_list]
    agent2_policies = [policy[1] for policy in policy_list]

    # Get action probability for each agent
    agent1_action1 = [action_prob[0] for action_prob in agent1_policies]
    agent1_action2 = [action_prob[1] for action_prob in agent1_policies]

    agent2_action1 = [action_prob[0] for action_prob in agent2_policies]
    agent2_action2 = [action_prob[1] for action_prob in agent2_policies]

    # Number of iterations
    iter_list = list(range(0, iter+1))

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.set_size_inches(8, 8, forward=True)
    fig.tight_layout(pad=5.0)

    # Plot Agent 1
    ax1.set(xlabel='Number of iterations', ylabel='Probability',title='Agent-1 policies through iterations')
    ax1.plot(iter_list, agent1_action1, linewidth=1.5, label="Probability of Action 1", color="#348ABD")
    ax1.plot(iter_list, agent1_action2, linewidth=1.5, label="Probability of Action 2", color="#A60628")
    ax1.legend()

    # Plot Agent 2
    ax2.set(xlabel='Number of iterations', ylabel='Probability', title='Agent-2 policies through iterations')
    ax2.plot(iter_list, agent2_action1, linewidth=1.5, label="Probability of Action 1", color="#7A68A6")
    ax2.plot(iter_list, agent2_action2, linewidth=1.5, label="Probability of Action 2", color="#467821")
    ax2.legend()

    plt.show()
