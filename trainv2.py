import os
import click
import time
import gym
import sys
import torch as T
from common import ClickPythonLiteralOption

from torch.utils.tensorboard import SummaryWriter

import numpy as np
from optproblems.cec2005 import *

from common.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper
from common.de_domain import DEFlattenedActionWrapper
from de_train_v2 import DEEnv

def pad_action(act, act_param, env_num, param_num):
    params = np.zeros((env_num,param_num,1), dtype=np.float32)
    params[np.arange(env_num), act, 0] = act_param
    return [(act[i], params[i]) for i in range(env_num)]


def evaluate(env, agent, episodes=1000):
    returns = []
    timesteps = []
    for _ in range(episodes):
        state, _ = env_list[0].reset()
        terminal = False
        t = 0
        total_reward = 0.
        while not terminal:
            t += 1
            state = np.array(state, dtype=np.float32, copy=False)
            act, act_param, all_action_parameters = agent.act(state)
            action = pad_action(act, act_param)
            (state, _), reward, terminal, _ = env_list[0].step(action)
            total_reward += reward
        timesteps.append(t)
        returns.append(total_reward)
    # return np.column_stack((returns, timesteps))
    return np.array(returns)


@click.command()
@click.option('--seed', default=1, help='Random seed.', type=int)
@click.option('--evaluation-episodes', default=1000, help='Episodes over which to evaluate after training.', type=int)
@click.option('--episodes', default=1000000, help='Number of epsiodes.', type=int)
@click.option('--batch-size', default=128, help='Minibatch size.', type=int)
@click.option('--gamma', default=0.99, help='Discount factor.', type=float)
@click.option('--inverting-gradients', default=True,
              help='Use inverting gradients scheme instead of squashing function.', type=bool)
@click.option('--initial-memory-threshold', default=1e5, help='Number of transitions required to start learning.',
              type=int)  # may have been running with 500??
@click.option('--use-ornstein-noise', default=True,
              help='Use Ornstein noise instead of epsilon-greedy with uniform random exploration.', type=bool)
@click.option('--replay-memory-size', default=1e6, help='Replay memory size in transitions.', type=int)
@click.option('--epsilon-steps', default=1e5, help='Number of episodes over which to linearly anneal epsilon.', type=int)
@click.option('--epsilon-final', default=0.1, help='Final epsilon value.', type=float)
@click.option('--tau-actor', default=0.1, help='Soft target network update averaging factor.', type=float)
@click.option('--tau-actor-param', default=0.001, help='Soft target network update averaging factor.', type=float)  # 0.001
@click.option('--learning-rate-actor', default=0.001, help="Actor network learning rate.", type=float) # 0.001/0.0001 learns faster but tableaus faster too
@click.option('--learning-rate-actor-param', default=0.0001, help="Critic network learning rate.", type=float)  # 0.00001
@click.option('--scale-actions', default=True, help="Scale actions.", type=bool)
@click.option('--initialise-params', default=True, help='Initialise action parameters.', type=bool)
@click.option('--clip-grad', default=1., help="Parameter gradient clipping limit.", type=float)
@click.option('--split', default=False, help='Separate action-parameter inputs.', type=bool)
@click.option('--multipass', default=True, help='Separate action-parameter inputs using multiple Q-network passes.', type=bool)
@click.option('--indexed', default=False, help='Indexed loss function.', type=bool)
@click.option('--weighted', default=False, help='Naive weighted loss function.', type=bool)
@click.option('--average', default=False, help='Average weighted loss function.', type=bool)
@click.option('--random-weighted', default=False, help='Randomly weighted loss function.', type=bool)
@click.option('--zero-index-gradients', default=False, help="Whether to zero all gradients for action-parameters not corresponding to the chosen action.", type=bool)
@click.option('--action-input-layer', default=0, help='Which layer to input action parameters.', type=int)
@click.option('--layers', default='[100,100,100,100]', help='Duplicate action-parameter inputs.', cls=ClickPythonLiteralOption)
@click.option('--save-freq', default=10, help='How often to save models (0 = never).', type=int)
@click.option('--save-dir', default="results/platform", help='Output directory.', type=str)
@click.option('--render-freq', default=100, help='How often to render / save frames of an episode.', type=int)
@click.option('--save-frames', default=False, help="Save render frames from the environment. Incompatible with visualise.", type=bool)
@click.option('--visualise', default=False, help="Render game states. Incompatible with save-frames.", type=bool)
@click.option('--title', default="MPDQNMulti", help="Prefix of output files", type=str)
@click.option('--reward-strategy', default="R1", help="Prefix of output files", type=str)
@click.option('--env-num', default=5, help="Number of environment solved per step", type=int)
def run(seed, episodes, evaluation_episodes, batch_size, gamma, inverting_gradients, initial_memory_threshold,
        replay_memory_size, epsilon_steps, tau_actor, tau_actor_param, use_ornstein_noise, learning_rate_actor,
        learning_rate_actor_param, epsilon_final, zero_index_gradients, initialise_params, scale_actions,
        clip_grad, split, indexed, layers, multipass, weighted, average, random_weighted, render_freq,
        save_freq, save_dir, save_frames, visualise, action_input_layer, title, reward_strategy="R1",
        env_num=5):

    # Tensorboard
    writer = SummaryWriter("runs/"+title+reward_strategy)

    if save_freq > 0 and save_dir:
        save_dir = os.path.join(save_dir, title + reward_strategy + "{}".format(str(seed)))
        os.makedirs(save_dir, exist_ok=True)


    dummy_env = DEEnv(reward_strategy=reward_strategy)
    initial_params_ = [0.5, 0.5, 0.5, 0.5]
    if scale_actions:
        for a in range(dummy_env.action_space.spaces[0].n):
            initial_params_[a] = 2. * (initial_params_[a] - dummy_env.action_space.spaces[1].spaces[a].low) / (
                        dummy_env.action_space.spaces[1].spaces[a].high - dummy_env.action_space.spaces[1].spaces[a].low) - 1.

    dummy_env = ScaledStateWrapper(dummy_env)
    dummy_env = DEFlattenedActionWrapper(dummy_env)
    if scale_actions:
        dummy_env = ScaledParameterisedActionWrapper(dummy_env)

    dir = os.path.join(save_dir,title)
    # env = Monitor(env, directory=os.path.join(dir,str(seed)), video_callable=False, write_upon_reset=False, force=True)
    # env_list[0].seed(seed)
    np.random.seed(seed)

    # print(dummy_env.observation_space)

    from agents.pdqn import PDQNAgent
    from agents.pdqn_split import SplitPDQNAgent
    from agents.pdqn_multipass import MultiPassPDQNAgent

    agent_class = MultiPassPDQNAgent
    agent = agent_class(
        dummy_env.observation_space.spaces[0], dummy_env.action_space,
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
        seed=seed,
        env_num=env_num)

    if initialise_params:
        initial_weights = np.zeros((dummy_env.action_space.spaces[0].n, dummy_env.observation_space.spaces[0].shape[0]))
        initial_bias = np.zeros(dummy_env.action_space.spaces[0].n)
        for a in range(dummy_env.action_space.spaces[0].n):
            initial_bias[a] = initial_params_[a]
        agent.set_action_parameter_passthrough_weights(initial_weights, initial_bias)

    print(agent)
    max_steps = int(1e4)
    total_reward = 0.
    returns = []
    start_time = time.time()
    best_average_stage_rewards = -999999999999.0
    stage_rewards = np.zeros(env_num, dtype=np.float32)

    # prepare the training function list
    func_choice = [unimodal.F1, unimodal.F2, unimodal.F5, basic_multimodal.F6, basic_multimodal.F8,
                            basic_multimodal.F10, basic_multimodal.F11, basic_multimodal.F12, expanded_multimodal.F13,
                            expanded_multimodal.F14, f15.F15, f19.F19, f20.F20, f21.F21, f22.F22, f24.F24]
    dim_choice = [10, 30]
    function_list = [func(dim) for func in func_choice for dim in dim_choice]
    num_function = len(function_list)
    random_func_idx = np.arange(num_function)
    current_episode = -1

    env_list = []
    for i in range(env_num):
        env = DEEnv(reward_strategy=reward_strategy)
        env_list += [env]
    initial_params_ = [0.5, 0.5, 0.5, 0.5]
    if scale_actions:
        for a in range(env_list[0].action_space.spaces[0].n):
            initial_params_[a] = 2. * (initial_params_[a] - env_list[0].action_space.spaces[1].spaces[a].low) / (
                        env_list[0].action_space.spaces[1].spaces[a].high - env_list[0].action_space.spaces[1].spaces[a].low) - 1.

    for i in range(env_num):
        env_list[i] = ScaledStateWrapper(env_list[i])
        env_list[i] = DEFlattenedActionWrapper(env_list[i])
        if scale_actions:
            env_list[i] = ScaledParameterisedActionWrapper(env_list[i])

    # for iteration
    states = np.zeros((env_num, 19), dtype=np.float32)
    next_states = np.zeros((env_num, 19), dtype=np.float32)
    rewards = np.zeros(env_num, dtype=np.float32)
    total_rewards = 0


    # save initial agent params and average stage rewards
    agent.save_models(os.path.join(save_dir, str(0)))
    writer.add_scalar("Average Stage Rewards", 0, 0)
    writer.flush()

    while current_episode < episodes:

        # Init environments
        terminal_flags = np.zeros(env_num, dtype=np.float32)
        env_rewards = np.zeros(env_num, dtype=np.float32)
        for env_idx in range(env_num):
            current_episode += 1
            if current_episode > 0 and current_episode%num_function==0:
                random_func_idx = np.random.choice(num_function, num_function, replace=False)
            state,_ = env_list[env_idx].reset_(func=function_list[random_func_idx[current_episode%num_function]])
            states[env_idx,:] = state
        act, act_param, all_action_parameters = agent.act(states)
        actions = pad_action(act, act_param, env_num, 4)

        total_Q_loss = 0
        agent.start_episode()
        for j in range(max_steps):

            # act per environment, and observe next states + rewards
            for env_idx in range(env_num):
                ret = env_list[env_idx].step(actions[env_idx])
                (next_state, steps), reward, terminal, _ = ret
                env_rewards[env_idx] += reward
                next_states[env_idx, :] = next_state
                rewards[env_idx] = reward
                terminal_flags[env_idx] = terminal

            # prepare next actions, and save them aka sarsa way
            next_act, next_act_param, next_all_action_parameters = agent.act(next_states)
            Q_loss = agent.step(states, (act, all_action_parameters), rewards, next_states,
                       (next_act, next_all_action_parameters), terminal_flags, steps)
            if Q_loss is not None:
                total_Q_loss += Q_loss

            # if one is done, all is done, break the optimization iteration
            if terminal_flags[0]:
                agent.end_episode()
                break

            # replace current action with next actions
            actions = pad_action(next_act, next_act_param, env_num, 4)


        # recap rewards, q_loss and network weights of each environments
        for env_idx in range(env_num):
            env_episode = current_episode-(env_num-env_idx)
            writer.add_scalar('total episode rewards', env_rewards[env_idx], env_episode)

        # recap average Q loss of 5 episodes
        writer.add_scalar('average episode q_loss', total_Q_loss/max_steps/env_num, current_episode)

        # recap average episode per stage (1 stage = 1 full function cycle/32 functions)
        for env_idx in range(env_num):
            env_episode = current_episode-(env_num-env_idx)
            if env_episode > 0 and env_episode%num_function==0:
                average_stage_rewards = total_rewards/num_function
                writer.add_scalar("Average Stage Rewards", average_stage_rewards, int(env_episode/num_function))
                if average_stage_rewards > best_average_stage_rewards:
                    best_average_stage_rewards = average_stage_rewards
                    agent.save_models(os.path.join(save_dir, str(i)))
                total_rewards = 0

            total_rewards += env_rewards[env_idx]

        # note weights l2 norm
        q_net_weights = 0.
        for param in agent.actor.parameters():
            q_net_weights += T.norm(param).item()

        writer.add_scalar('Q-network Weights L2 norm', q_net_weights, current_episode)

        actor_weights = 0.
        for param in agent.actor_param.parameters():
            actor_weights += T.norm(param).item()

        writer.add_scalar('Actor Weights L2 norm', actor_weights, current_episode)
        writer.flush()

if __name__ == '__main__':
    T.set_num_threads(6)
    run()
