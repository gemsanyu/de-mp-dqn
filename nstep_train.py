import os
import click
import time
from common import ClickPythonLiteralOption
import gym
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter

from agents.utils import soft_update_target_network
from common.wrappers import ScaledStateWrapper, ScaledParameterisedActionWrapper
from common.de_domain import DEFlattenedActionWrapper
from de_train import DEEnv

def pad_action(act, act_param):
    params = [np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32),
              np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32)]
    params[act][:] = act_param
    return (act, params)

@click.command()
@click.option('--seed', default=1, help='Random seed.', type=int)
@click.option('--episodes', default=1000000, help='Number of epsiodes.', type=int)
@click.option('--evaluation-episodes', default=1000, help='Episodes over which to evaluate after training.', type=int)
@click.option('--batch-size', default=256, help='Minibatch size.', type=int)
@click.option('--gamma', default=0.9, help='Discount factor.', type=float)
@click.option('--update-ratio', default=0.2, help='Ratio of updates to samples.', type=float)
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

    if save_freq > 0 and save_dir:
        save_dir = os.path.join(save_dir, title + "{}".format(str(seed)) + reward_strategy)
        os.makedirs(save_dir, exist_ok=True)

   # Tensorboard
    writer = SummaryWriter("runs/"+title+reward_strategy)

    env = DEEnv(reward_definition=reward_strategy)
    initial_params_ = [0.5, 0.5, 0.5, 0.5]
    if scale_actions:
        for a in range(env.action_space.spaces[0].n):
            initial_params_[a] = 2. * (initial_params_[a] - env.action_space.spaces[1].spaces[a].low) / (
                        env.action_space.spaces[1].spaces[a].high - env.action_space.spaces[1].spaces[a].low) - 1.

    env = ScaledStateWrapper(env)
    env = DEFlattenedActionWrapper(env)
    if scale_actions:
        env = ScaledParameterisedActionWrapper(env)

    dir = os.path.join(save_dir, title+reward_strategy)
    np.random.seed(seed)

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
    if initialise_params:
        initial_weights = np.zeros((env.action_space.spaces[0].n, env.observation_space.spaces[0].shape[0]))
        initial_bias = np.zeros(env.action_space.spaces[0].n)
        for a in range(env.action_space.spaces[0].n):
            initial_bias[a] = initial_params_[a]
        agent.set_action_parameter_passthrough_weights(initial_weights, initial_bias)

    print(agent)
    network_trainable_parameters = sum(p.numel() for p in agent.actor.parameters() if p.requires_grad)
    network_trainable_parameters += sum(p.numel() for p in agent.actor_param.parameters() if p.requires_grad)
    print("Total Trainable Network Parameters: %d" % network_trainable_parameters)
    max_steps = 10000
    total_reward = 0.
    returns = []
    timesteps = []
    goals = []
    start_time_train = time.time()

    # load models
    # if target networks are not saved, then they have to be approximated by simple liner interpolation
    last_episode = save_freq
    load_dir = os.path.join(save_dir, str(last_episode))
    while os.path.exists(load_dir+"_actor.pt"):
        print("LOADING NETWORKS FROM EPISODE-"+str(last_episode))
        #Q network (critic)
        old_params = agent.actor.parameters()
        agent.actor.load_state_dict(torch.load(load_dir + '_actor.pt', map_location='cpu'))
        for target_param, param, old_param in zip(agent.actor_target.parameters(),
                                                  agent.actor.parameters(),
                                                  old_params):
            # iteratively update the target networks
            # with the linear interpolation of the current primary network + next snapshot's primary network
            primary_diff = param.data - old_param.data
            for i in range(1, save_freq):
                target_param.data.copy_(tau_actor * (param.data + (float(i)/float(save_freq))*primary_diff)
                                        (1.0 - tau_actor) * target_param.data)

        # parameter network (actor)
        old_params = agent.actor_param.parameters
        agent.actor_param.load_state_dict(torch.load(load_dir + '_actor_param.pt', map_location='cpu'))
        for target_param, param, old_param in zip(agent.actor_param_target.parameters(),
                                                  agent.actor_param.parameters(),
                                                  old_params):
            # iteratively update the target networks
            # with the linear interpolation of the current primary network + next snapshot's primary network
            primary_diff = param.data - old_param.data
            for i in range(1, save_freq):
                target_param.data.copy_(tau_actor * (param.data + (float(i)/float(save_freq))*primary_diff)
                                        (1.0 - tau_actor) * target_param.data)

        last_episode = last_episode + save_freq
        load_dir = os.path.join(save_dir, str(last_episode))

    # while os.path.exists(load_dir):
    #     next_episode = last_episode + save_freq
    #     next_load_dir = os.path.join(save_dir, str(next_episode))
    #     if not os.path.exists(next_load_dir):
    #         break
    #         self.actor_tmp.load_state_dict(torch.load(next_load_dir + '_actor.pt', map_location='cpu'))
    #         self.actor_param_tmp.load_state_dict(torch.load(next_load_dir + '_actor_param.pt', map_location='cpu'))
    #
    #         # add to target network
    #         for actor_target_p, actor_p, actor_tmp_p in zip(self.actor_target.parameters(),
    #                                                         self.actor.parameters(),
    #                                                         self.actor_tmp.parameters()):
    #             for i in range(save_freq):
    #                 actor_target_p.data.copy_(
    #                     self.tau_actor*(actor_p.data+(1.0/float(i))*(actor_tmp_p.data-actor_p.data)) +
    #                     (1-self.tau_actor)*actor_target_p)
    #
    #         for param_target_p, param_p, param_tmp_p in zip(self.actor_param_target.parameters(),
    #                                                         self.actor_param.parameters(),
    #                                                         self.actor_param_tmp.parameters()):
    #             for i in range(save_freq):
    #                 param_target_p.data.copy_(
    #                     self.tau_actor*(param_p.data+(1.0/float(i))*(param_tmp_p.data-actor_p.data)) +
    #                     (1-self.tau_actor)*actor_target_p)
    # # Models are loaded #
    last_episode = last_episode - save_freq
    for i in range(last_episode, episodes):
        if save_freq > 0 and save_dir and i % save_freq == 0:
            agent.save_models(os.path.join(save_dir, str(i)))
        info = {'status': "NOT_SET"}
        state, _ = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)

        act, act_param, all_action_parameters = agent.act(state)
        action = pad_action(act, act_param)
        episode_reward = 0.
        agent.start_episode()
        transitions = []
        for j in range(max_steps):
            (next_state,steps), reward, terminal, info = env.step(action)
            next_state = np.array(next_state, dtype=np.float32, copy=False)
            # status = info['status']
            # if status != 'IN_GAME':
            #     print(status)

            next_act, next_act_param, next_all_action_parameters = agent.act(next_state)
            next_action = pad_action(next_act, next_act_param)
            transitions.append([state, np.concatenate(([act], all_action_parameters.data)).ravel(), reward,
                                next_state, np.concatenate(([next_act],
                                                            next_all_action_parameters.data)).ravel(), terminal])

            act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
            action = next_action
            state = next_state
            episode_reward += reward
            #env.render()

            if terminal:
                break
        agent.end_episode()

        # calculate n-step returns
        n_step_returns = compute_n_step_returns(transitions, gamma)
        for t, nsr in zip(transitions, n_step_returns):
            t.append(nsr)
            agent.replay_memory.append(state=t[0], action=t[1], reward=t[2], next_state=t[3], next_action=t[4],
                                       terminal=t[5], time_steps=None, n_step_return=nsr)

        n_updates = int(update_ratio*j)
        total_q_loss = 0
        for _ in range(n_updates):
            q_loss = agent._optimize_td_loss()
            if q_loss is not None:
                total_q_loss += q_loss

        writer.add_scalar('total episode rewards', episode_reward, i)
        writer.add_scalar('average episode q_loss', total_q_loss/max_steps, i)

        # note weights l2 norm
        q_net_weights = 0.
        for param in agent.actor.parameters():
            q_net_weights += torch.norm(param).item()

        writer.add_scalar('Q-network Weights L2 norm', q_net_weights, i)

        actor_weights = 0.
        for param in agent.actor_param.parameters():
            actor_weights += torch.norm(param).item()

        writer.add_scalar('Actor Weights L2 norm', actor_weights, i)
        writer.flush()


        returns.append(episode_reward)
        timesteps.append(j)
        # goals.append(info['status'] == 'GOAL')

        total_reward += episode_reward
        if i % 100 == 0:
            print('{0:5s} R:{1:.4f} r100:{2:.4f}'.format(str(i + 1), total_reward / (i + 1), np.array(returns[-100:]).mean()))
    end_time_train = time.time()
    if save_freq > 0 and save_dir:
        agent.save_models(os.path.join(save_dir, str(i)))

    # returns = env.get_episode_rewards()
    # np.save(os.path.join(dir, title + "{}".format(str(seed))), np.column_stack((returns, timesteps, goals)))

    # if evaluation_episodes > 0:
    #     print("Evaluating agent over {} episodes".format(evaluation_episodes))
    #     agent.epsilon_final = 0.
    #     agent.epsilon = 0.
    #     agent.noise = None
    #     agent.actor.eval()
    #     agent.actor_param.eval()
    #     start_time_eval = time.time()
    #     evaluation_results = evaluate(env, agent, evaluation_episodes)  # returns, timesteps, goals
    #     end_time_eval = time.time()
    #     print("Ave. evaluation return =", sum(evaluation_results[:, 0]) / evaluation_results.shape[0])
    #     print("Ave. timesteps =", sum(evaluation_results[:, 1]) / evaluation_results.shape[0])
    #     goal_timesteps = evaluation_results[:, 1][evaluation_results[:, 2] == 1]
    #     if len(goal_timesteps) > 0:
    #         print("Ave. timesteps per goal =", sum(goal_timesteps) / evaluation_results.shape[0])
    #     else:
    #         print("Ave. timesteps per goal =", sum(goal_timesteps) / evaluation_results.shape[0])
    #     print("Ave. goal prob. =", sum(evaluation_results[:, 2]) / evaluation_results.shape[0])
    #     np.save(os.path.join(dir, title + "{}e".format(str(seed))), evaluation_results)
    #     print("Evaluation time: %.2f seconds" % (end_time_eval - start_time_eval))
    # print("Training time: %.2f seconds" % (end_time_train - start_time_train))

    print(agent)
    # env.close()


def compute_n_step_returns(episode_transitions, gamma):
    n = len(episode_transitions)
    n_step_returns = np.zeros((n,))
    n_step_returns[n-1] = episode_transitions[n-1][2]  # Q-value is just the final reward
    for i in range(n-2, 0, -1):
        reward = episode_transitions[i][2]
        target = n_step_returns[i + 1]
        n_step_returns[i] = reward + gamma * target
    return n_step_returns


if __name__ == '__main__':
    torch.set_num_threads(6)
    run()
