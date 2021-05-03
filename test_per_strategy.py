import os
import math
import click
import time
import gym
import sys
from common import ClickPythonLiteralOption

import torch
import numpy as np
from optproblems.cec2005 import *

from common.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper
from common.de_domain import DEFlattenedActionWrapper
from agents.pdqn_multipass import MultiPassPDQNAgent
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
        fitness.append(math.fabs(env.best_so_far-env.best_value))
        returns.append(total_reward)
    # return np.column_stack((returns, timesteps))
    fitness = np.array(fitness)
    # print(env.fun, env.dim, np.mean(fitness), np.std(fitness))
    return float(np.mean(fitness)), float(np.std(fitness))


@click.command()
@click.option('--seed', default=1, help='Random seed.', type=int)
@click.option('--evaluation-episodes', default=1000, help='Episodes over which to evaluate after training.', type=int)
@click.option('--episodes', default=200, help='Number of epsiodes.', type=int)
@click.option('--batch-size', default=64, help='Minibatch size.', type=int)
@click.option('--gamma', default=0.9, help='Discount factor.', type=float)
@click.option('--inverting-gradients', default=True,
              help='Use inverting gradients scheme instead of squashing function.', type=bool)
@click.option('--initial-memory-threshold', default=500, help='Number of transitions required to start learning.',
              type=int)  # may have been running with 500??
@click.option('--use-ornstein-noise', default=False,
              help='Use Ornstein noise instead of epsilon-greedy with uniform random exploration.', type=bool)
@click.option('--replay-memory-size', default=1e5, help='Replay memory size in transitions.', type=int)
@click.option('--epsilon-steps', default=0, help='Number of episodes over which to linearly anneal epsilon.', type=int)
@click.option('--epsilon-final', default=0, help='Final epsilon value.', type=float)
@click.option('--tau-actor', default=0.1, help='Soft target network update averaging factor.', type=float)
@click.option('--tau-actor-param', default=0.001, help='Soft target network update averaging factor.', type=float)  # 0.001
@click.option('--learning-rate-actor', default=0.001, help="Actor network learning rate.", type=float) # 0.001/0.0001 learns faster but tableaus faster too
@click.option('--learning-rate-actor-param', default=0.0001, help="Critic network learning rate.", type=float)  # 0.00001
@click.option('--scale-actions', default=True, help="Scale actions.", type=bool)
@click.option('--initialise-params', default=True, help='Initialise action parameters.', type=bool)
@click.option('--clip-grad', default=10., help="Parameter gradient clipping limit.", type=float)
@click.option('--split', default=False, help='Separate action-parameter inputs.', type=bool)
@click.option('--multipass', default=True, help='Separate action-parameter inputs using multiple Q-network passes.', type=bool)
@click.option('--indexed', default=False, help='Indexed loss function.', type=bool)
@click.option('--weighted', default=False, help='Naive weighted loss function.', type=bool)
@click.option('--average', default=False, help='Average weighted loss function.', type=bool)
@click.option('--random-weighted', default=False, help='Randomly weighted loss function.', type=bool)
@click.option('--zero-index-gradients', default=False, help="Whether to zero all gradients for action-parameters not corresponding to the chosen action.", type=bool)
@click.option('--action-input-layer', default=0, help='Which layer to input action parameters.', type=int)
@click.option('--layers', default='[100,100,100,100]', help='Duplicate action-parameter inputs.', cls=ClickPythonLiteralOption)
@click.option('--save-freq', default=1, help='How often to save models (0 = never).', type=int)
@click.option('--save-dir', default="results/platform", help='Output directory.', type=str)
@click.option('--render-freq', default=100, help='How often to render / save frames of an episode.', type=int)
@click.option('--save-frames', default=False, help="Save render frames from the environment. Incompatible with visualise.", type=bool)
@click.option('--visualise', default=False, help="Render game states. Incompatible with save-frames.", type=bool)
@click.option('--title', default="DE-MPDQN", help="Prefix of output files", type=str)
@click.option('--reward-strategy', default="R1", help="Reward Strategy of the environment", type=str)
def run(seed, episodes, evaluation_episodes, batch_size, gamma, inverting_gradients, initial_memory_threshold,
        replay_memory_size, epsilon_steps, tau_actor, tau_actor_param, use_ornstein_noise, learning_rate_actor,
        learning_rate_actor_param, epsilon_final, zero_index_gradients, initialise_params, scale_actions,
        clip_grad, split, indexed, layers, multipass, weighted, average, random_weighted, render_freq,
        save_freq, save_dir, save_frames, visualise, action_input_layer, title, reward_strategy="R1"):


    np.random.seed()
    dir = os.path.join(save_dir,title)
    save_dir = os.path.join(save_dir, title + reward_strategy + "{}".format(str(seed)))

    agent_class = MultiPassPDQNAgent

    start_time = time.time()

    max_runs = 25
    dims = [10, 30]
    func_select = [unimodal.F3, basic_multimodal.F9, f16.F16, f18.F18, f23.F23]

    # Init agent with dummy env, because so many wrappers, i don't really know what the feature and action space
    # structured

    # DUMMY ENV
    d = 10
    fun = unimodal.F3(d)
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

    # Init agent with dummy ENV
    agent = agent_class(
        env.observation_space.spaces[0], env.action_space,
        batch_size=batch_size,
        learning_rate_actor=learning_rate_actor,
        learning_rate_actor_param=learning_rate_actor_param,
        epsilon_steps=epsilon_steps,
        gamma=gamma,
        tau_actor=tau_actor,
        tau_actor_param=tau_actor_param,
        clip_grad=clip_grad,
        indexed=indexed,
        weighted=weighted,
        average=average,
        random_weighted=random_weighted,
        initial_memory_threshold=initial_memory_threshold,
        use_ornstein_noise=use_ornstein_noise,
        replay_memory_size=replay_memory_size,
        epsilon_final=epsilon_final,
        inverting_gradients=inverting_gradients,
        actor_kwargs={'hidden_layers': layers,
                      'action_input_layer': action_input_layer,},
        actor_param_kwargs={'hidden_layers': layers,
                            'squashing_function': False,
                            'output_layer_init_std': 0.0001,},
        zero_index_gradients=zero_index_gradients,
        seed=seed)
    agent.variances = 0
    agent.epsilon = 0.
    agent.noise = None
    agent.actor.eval()
    agent.actor_param.eval()

    #preparing output files
    mean_result_file = open(reward_strategy+"mean.csv", "w+")
    mean_result_file.write("RUN,F3-10,F3-10,F3-10,F3-10,F3-10,F3-10,F3-10,F16-30,F18-30,F23-30\n")

    std_result_file = open(reward_strategy+"std.csv", "w+")
    std_result_file.write("RUN,F3-10,F3-10,F3-10,F3-10,F3-10,F3-10,F3-10,F16-30,F18-30,F23-30\n")

    epoch = 10
    load_dir = os.path.join(save_dir, str(epoch))
    while os.path.exists(load_dir+"_actor.pt"):
        agent.load_models(load_dir)
        print(load_dir)
        mean_new_line = str(epoch)
        std_new_line = str(epoch)
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
                mean_err, std_err = evaluate(env, agent, max_runs)
                mean_new_line = mean_new_line + "," + str(mean_err)
                std_new_line = std_new_line + "," + str(std_err)

        mean_result_file.write(mean_new_line+"\n")
        std_result_file.write(std_new_line+"\n")
        mean_result_file.flush()
        std_result_file.flush()
        epoch += 10
        load_dir = os.path.join(save_dir, str(epoch))

    mean_result_file.close()
    std_result_file.close()

if __name__ == '__main__':
    torch.set_num_threads(3)
    run()
