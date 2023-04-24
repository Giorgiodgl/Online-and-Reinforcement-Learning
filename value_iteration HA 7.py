import numpy as np


def expected_state_action_reward(env, s, a, V):
    """
    Computes the expected reward of taking action a in state s, according to the given environment.
    I.e. compute:
        r(s, a) + sum_{s'} p(s'|s, a) * V(s')
    :param env: an MDP environment
    :param s: a state
    :param a: an action
    :param V: a value function
    :return: the expected reward of taking action a in state s, according to the given environment
    """
    return env.R[s, a] + np.sum([env.P[s, a, s_] * V[s_] for s_ in range(env.nS)])


def value_to_policy(V, env):
    """
    Computes the gain and bias of a policy based on a value function.
    :param V: a value function
    :param env: an MDP environment
    :return: an optimal policy based on V
    """
    policy = np.zeros(env.nS, dtype=int)
    for s in range(env.nS):
        policy[s] = np.argmax([expected_state_action_reward(env, s, a, V) for a in range(env.nA)])

    return policy


def span_semi_norm(f):
    """
    Computes the span of a semi-norm.
    :param f: a function f: X -> R^S
    :return: the span of f
    """
    return np.max(f) - np.min(f)


def value_iteration_average_reward(env, epsilon=10 ** -6):
    """
    Performs value iteration on the given environment, for an average reward objective.
    :param env: an MDP environment
    :param epsilon: the precision required for the value function
    :return:
        V: the value function
        policy: an optimal policy based on V
        gain: the gain of 'policy'
        bias: an associated bias function
        n_iter = the number of iterations required to converge
    """

    # Initialization
    V0 = np.zeros(env.nS)
    V1 = np.zeros(env.nS)
    n_iter = 0

    # Iterate until convergence
    while True:
        n_iter += 1
        V0 = V1.copy()
        for s in range(env.nS):
            V1[s] = np.max([expected_state_action_reward(env, s, a, V0) for a in range(env.nA)])

        if span_semi_norm(V1 - V0) < epsilon:
            break

    # Compute policy, gain and bias
    V = V1  # Final value function
    policy = value_to_policy(V, env)
    gain = (np.max(V1 - V0) + np.min(V1 - V0)) / 2
    bias = V - np.min(V)

    return V, policy, gain, bias, n_iter
