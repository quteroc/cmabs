import numpy as np


def explore_and_commit(env, explore_steps = 50, iters = 200):
    clicks = np.zeros((env.n_states, env.n_actions))
    views = np.zeros((env.n_states, env.n_actions))
    Q = np.zeros((env.n_states, env.n_actions))
    Qs = []
    total_reward = 0.
    regret = 0.

    # Explore
    for i in range(explore_steps):
        state = env.observe()
        action = np.random.choice(env.n_actions)
        click = env.step(action)
        views[state, action] = views[state, action] + 1
        clicks[state, action] = clicks[state, action] + click
        Q[state, action] = clicks[state, action] / views[state, action]
        total_reward += click
        best_action = env.CTR[state,:].argmax()
        regret += env.CTR[state,best_action] - env.CTR[state,action]
        Qs.append(Q.copy())

    # Commit
    for i in range(iters-explore_steps):
        state = env.observe()
        action = Q[state,:].argmax()
        click = env.step(action)
        total_reward = total_reward + click
        regret += env.CTR[state,:].max() - env.CTR[state,action]

    return Qs, total_reward, regret


def epsilon_greedy(env, epsilon = 0.1, null_epsilon_after = 50, iters = 200):
    clicks = np.zeros((env.n_states, env.n_actions))
    views = np.zeros((env.n_states, env.n_actions))
    Q = np.zeros((env.n_states, env.n_actions))
    Qs = []
    total_reward = 0.
    regret = 0.

    # "Explore" (epsilon is non-zero)
    for i in range(null_epsilon_after):
        state = env.observe()
        if np.random.rand() < epsilon:
            action = np.random.choice(env.n_actions)
        else:
            action = Q[state, :].argmax()
        click = env.step(action)
        views[state, action] = views[state, action] + 1
        clicks[state, action] = clicks[state, action] + click
        Q[state, action] = clicks[state, action] / views[state, action]
        total_reward = total_reward + click
        best_action = env.CTR[state,:].argmax()
        regret += env.CTR[state,best_action] - env.CTR[state,action]
        Qs.append(Q.copy())

    # "Commit" (epsilon set to 0)
    for i in range(iters-null_epsilon_after):
        state = env.observe()
        action = Q[state,:].argmax()
        click = env.step(action)
        total_reward = total_reward + click
        best_action = env.CTR[state,:].argmax()
        regret += env.CTR[state,best_action] - env.CTR[state,action]

    return Qs, total_reward, regret