import os
import click
import time
from common import ClickPythonLiteralOption
import gym
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
from optproblems.cec2005 import *

from common.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper
from common.de_domain import DEFlattenedActionWrapper
from de_test import DEEnv

def pad_action(act, act_param):
    params = [np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32),
              np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32)]
    params[act][:] = act_param
    return (act, params)

def evaluate(env, agent, episodes=25):
    returns = []
    timesteps = []
    fitness = []
    for _ in range(episodes):
        state, _ = env.reset()
        terminal = False
        t = 0
        total_reward = 0.
        while not terminal:
            t += 1
            state = np.array(state, dtype=np.float32, copy=False)
            act, act_param, all_action_parameters = agent.act(state)
            action = pad_action(act, act_param)
            (state, _), reward, terminal, _ = env.step(action)
            total_reward += reward
            # print(action, env.best_so_far)
        timesteps.append(t)
        fitness.append(env.best_so_far)
        returns.append(total_reward)
    # return np.column_stack((returns, timesteps))
    fitness = np.array(fitness)
    print(env.fun, env.dim, np.mean(fitness), np.std(fitness))
    return np.array(returns)

@click.command()
@click.option('--seed', default=1, help='Random seed.', type=int)
@click.option('--episodes', default=200000, help='Number of epsiodes.', type=int)
@click.option('--evaluation-episodes', default=1000, help='Episodes over which to evaluate after training.', type=int)
@click.option('--batch-size', default=256, help='Minibatch size.', type=int)
@click.option('--gamma', default=0.9, help='Discount factor.', type=float)
@click.option('--update-ratio', default=0.1, help='Ratio of updates to samples.', type=float)
@click.option('--inverting-gradients', default=True,
              help='Use inverting gradients scheme instead of squashing function.', type=bool)
@click.option('--initial-memory-threshold', default=100000, help='Number of transitions required to start learning.',
              type=int)
@click.option('--use-ornstein-noise', default=False,
              help='Use Ornstein noise instead of epsilon-greedy with uniform random exploration.', type=bool)
@click.option('--replay-memory-size', default=1000000, help='Replay memory size in transitions.', type=int) # 500000
@click.option('--epsilon-steps', default=100000, help='Number of episodes over which to linearly anneal epsilon.', type=int)
@click.option('--epsilon-final', default=0.1, help='Final epsilon value.', type=float)
@click.option('--tau-actor', default=0.01, help='Soft target network update averaging factor.', type=float)
@click.option('--tau-actor-param', default=0.001, help='Soft target network update averaging factor.', type=float)  # 0.001
@click.option('--learning-rate-actor', default=0.001, help="Actor network learning rate.", type=float)
@click.option('--learning-rate-actor-param', default=0.00001, help="Critic network learning rate.", type=float)
@click.option('--clip-grad', default=10., help="Gradient clipping.", type=float)  # 1 better than 10.
@click.option('--beta', default=0.2, help='Averaging factor for on-policy and off-policy targets.', type=float)  # 0.5
@click.option('--scale-actions', default=True, help="Scale actions.", type=bool)
@click.option('--split', default=False, help='Separate action-parameter inputs.', type=bool)
@click.option('--multipass', default=True, help='Separate action-parameter inputs using multiple Q-network passes.', type=bool)
@click.option('--indexed', default=False, help='Indexed loss function.', type=bool)
@click.option('--weighted', default=False, help='Naive weighted loss function.', type=bool)
@click.option('--average', default=False, help='Average weighted loss function.', type=bool)
@click.option('--random-weighted', default=False, help='Randomly weighted loss function.', type=bool)
@click.option('--zero-index-gradients', default=False, help="Whether to zero all gradients for action-parameters not corresponding to the chosen action.", type=bool)
@click.option('--action-input-layer', default=0, help='Which layer to input action parameters at when using split Q-networks.', type=int)
@click.option('--layers', default="[100,100,100,100]", help='Duplicate action-parameter inputs.', cls=ClickPythonLiteralOption)
@click.option('--save-freq', default=10, help='How often to save models (0 = never).', type=int)
@click.option('--save-dir', default="results/nstep", help='Output directory.', type=str)
@click.option('--title', default="N-STEP", help="Prefix of output files", type=str)
@click.option('--initialise-params', default=True, help='Initialise action parameters.', type=bool)
@click.option('--reward-strategy', default="R1", help="Prefix of output files", type=str)
def run(seed, episodes, batch_size, gamma, inverting_gradients, initial_memory_threshold, replay_memory_size,
        epsilon_steps, tau_actor, tau_actor_param, use_ornstein_noise, learning_rate_actor, learning_rate_actor_param, title, epsilon_final,
        clip_grad, beta, scale_actions, split, indexed, zero_index_gradients, action_input_layer,
        evaluation_episodes, multipass, weighted, average, random_weighted, update_ratio,
        save_freq, save_dir, layers, initialise_params, reward_strategy):

    np.random.seed()

    from agents.pdqn_nstep import PDQNNStepAgent
    from agents.pdqn_split_nstep import PDQNNStepSplitAgent
    from agents.pdqn_multipass_nstep import MultiPassPDQNNStepAgent
    assert not (split and multipass)
    agent_class = PDQNNStepAgent
    if split:
        agent_class = PDQNNStepSplitAgent
    elif multipass:
        agent_class = MultiPassPDQNNStepAgent
    assert action_input_layer >= 0
    if action_input_layer > 0:
        assert split


    dir = os.path.join(save_dir,title)
    save_dir = os.path.join(save_dir, title + "{}".format(str(seed) + reward_strategy))
    max_runs = 1
    dims = [10, 30]
    # func_select = [f18.F18, f23.F23]
    func_select = [unimodal.F3, basic_multimodal.F9, f16.F16, f18.F18, f23.F23]
    for d in dims:
        for f in func_select:
            fun = f(d)
            lbounds = fun.min_bounds; lbounds = np.array(lbounds); #print(lbounds)
            ubounds = fun.max_bounds; ubounds = np.array(ubounds); #print(ubounds)
            opti = fun.get_optimal_solutions()
            for o in opti:
                print(o.phenome, o.objective_values)
            sol = np.copy(o.phenome)
            best_value = fun.objective_function(sol)

            env = DEEnv(fun, lbounds, ubounds, d, best_value)
            env = ScaledStateWrapper(env)
            env = DEFlattenedActionWrapper(env)
            if scale_actions:
                env = ScaledParameterisedActionWrapper(env)

            agent = agent_class(
               env.observation_space.spaces[0], env.action_space,
               actor_kwargs={"hidden_layers": layers,
                             'action_input_layer': action_input_layer,
                             'activation': "relu"},
               actor_param_kwargs={"hidden_layers": layers,
                                    'activation': "relu",
                                   'output_layer_init_std': 0.0001},
               batch_size=batch_size,
               learning_rate_actor=learning_rate_actor,  # 0.0001
               learning_rate_actor_param=learning_rate_actor_param,  # 0.001
               epsilon_steps=epsilon_steps,
               epsilon_final=epsilon_final,
               gamma=gamma,  # 0.99
               tau_actor=tau_actor,
               tau_actor_param=tau_actor_param,
               clip_grad=clip_grad,
               beta=beta,
               indexed=indexed,
               weighted=weighted,
               average=average,
               random_weighted=random_weighted,
               initial_memory_threshold=initial_memory_threshold,
               use_ornstein_noise=use_ornstein_noise,
               replay_memory_size=replay_memory_size,
               inverting_gradients=inverting_gradients,
               zero_index_gradients=zero_index_gradients,
               seed=seed)
            load_dir = os.path.join(save_dir, str(32400))
            agent.load_models(load_dir)
            agent.variances = 0
            agent.epsilon = 0.
            agent.noise = None
            agent.actor.eval()
            agent.actor_param.eval()

            # agent.discrete_agent.epsilon = 0.
            # agent.discrete_agent.temperature = 0.
            evaluate(env, agent, max_runs)

    # returns = env.get_episode_rewards()
    # np.save(os.path.join(dir, title + "{}".format(str(seed))),returns)

    # print(agent)
    # env.close()



if __name__ == '__main__':
    # torch.set_num_threads(6)
    run()
