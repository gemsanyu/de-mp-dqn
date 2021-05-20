import gym
import numpy as np
from optproblems.cec2005 import *
from gym import spaces

import math


def rand1(population, samples, scale, best, i): # DE/rand/1
    r0, r1, r2 = samples[:3]
    return population[r0] + scale * (population[r1] - population[r2])


def rand2(population, samples, scale, best, i): # DE/rand/2
    r0, r1, r2, r3, r4 = samples[:5]
    return population[r0] + scale * (population[r1] - population[r2] + population[r3] - population[r4])


def rand_to_best2(population, samples, scale, best, i): # DE/rand-to-best/2
    r0, r1, r2, r3, r4 = samples[:5]
    return population[r0] + \
           scale * (population[best] - population[r0] + population[r1] - population[r2] + population[r3] - population[r4])


def current_to_rand1(population, samples, scale, best, i): # DE/current-to-rand/1
    r0, r1, r2 = samples[:3]
    return population[i] + scale * (population[r0] - population[i] + population[r1] - population[r2])


# somehow a little faster than np.linalg.norm or scipy.distance
def get_distance(a, b):
    dist = [(x-y)**2 for x,y in zip(a,b)]
    return math.sqrt(sum(dist))


mutation_operators = [rand1, rand2, rand_to_best2, current_to_rand1]


class DEEnv(gym.Env):
    def __init__(self, random_seed=0, reward_strategy="R1"):
        super(DEEnv, self).__init__()

        self.feature_size = 19
        self.n_ops=len(mutation_operators)
        self.action_space = spaces.Tuple((
            spaces.Discrete(self.n_ops),  # actions
            spaces.Tuple(  # parameters
                tuple(spaces.Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
                      for _ in range(self.n_ops))
            )
        ))
        self.observation_space = spaces.Tuple((
            spaces.Box(low=-np.inf, high=np.inf, shape=(self.feature_size,), dtype=np.float32),
            spaces.Discrete(10000),  # steps (200 limit is an estimate)
        ))
        np.random.seed(random_seed)
        self.CR = 1
        self.NP = 100
        self.reward_strategy = reward_strategy
        self.max_fe = 10000

        # running variables
        self.n_fe = 0
        self.gen = 0
        self.func = None
        self.dim = 0
        self.min_bounds = None
        self.max_bounds = None
        self.current_best_value = 0
        self.current_best_idx = 0
        self.current_worst_value = 0
        self.real_best_value = 0
        self.X = None
        self.F = None
        self.current_idx = 0
        self.max_dist = 0
        self.max_std = 0
        self.stag_count = 0
        self.random_idx = None

        # training function list
        func_choice = [unimodal.F1, unimodal.F2, unimodal.F5, basic_multimodal.F6, basic_multimodal.F8,
                            basic_multimodal.F10, basic_multimodal.F11, basic_multimodal.F12, expanded_multimodal.F13,
                            expanded_multimodal.F14, f15.F15, f19.F19, f20.F20, f21.F21, f22.F22, f24.F24]
        dim_choice = [10, 30]
        self.function_list = [func(dim) for func in func_choice for dim in dim_choice]
        self.num_function = len(self.function_list)
        self.randomized_func_idx = np.arange(len(self.function_list))
        self.current_function_idx = -1

        # save optimal value of each function
        self.func_best_value_list = []
        for func in self.function_list:
            opti = func.get_optimal_solutions()
            solution = np.copy(opti[0].phenome)
            best_value = func.objective_function(solution)
            self.func_best_value_list += [best_value]

        # save min

    def reset_(self, func=None, is_test=False):

        # set function and its best values
        if not is_test:
            self.current_function_idx += 1
            if self.current_function_idx == self.num_function:
                self.current_function_idx = 0
                self.randomized_func_idx = np.random.choice(self.num_function, self.num_function, replace=False)
            self.func = self.function_list[self.randomized_func_idx[self.current_function_idx]]
            self.real_best_value = self.func_best_value_list[self.randomized_func_idx[self.current_function_idx]]
        else:
            if func is None:
                print("ERROR, reset_: new_func cannot")
                return None, None
            self.func = func
            opti = func.get_optimal_solutions()
            solution = np.copy(opti[0].phenome)
            self.real_best_value = func.objective_function(solution)

        # set bounds
        self.min_bounds = np.array(self.func.min_bounds)
        self.max_bounds = np.array(self.func.max_bounds)
        self.dim = len(self.min_bounds)
        return self.reset()

    # reset_ is changing function too, while reset() is just resetting the number of FE, for the same function
    # and also reinitializing population
    def reset(self):
        self.n_fe = self.NP
        self.gen = 0
        self.current_best_value = 0
        self.current_best_idx = 0

        self.X = self.min_bounds + (self.max_bounds-self.min_bounds)*(np.random.rand(self.NP, self.dim))
        self.F = np.array([self.func(x) for x in self.X])
        self.current_best_value = np.min(self.F)
        self.current_best_idx = np.argmin(self.F)
        self.current_worst_value = np.max(self.F)
        self.current_idx = 0

        self.max_dist = get_distance(self.min_bounds, self.max_bounds)
        self.max_std = np.std((np.repeat(self.current_best_value, self.NP/2), np.repeat(self.current_worst_value, self.NP/2)))
        self.stag_count = 0
        self.random_idx = np.random.choice(self.NP, size=5, replace=False).astype(int)
        feature = self.get_feature(self.current_idx, self.random_idx)
        # initial feature or observation, no reward or flag is returned
        return feature, None

    def get_feature(self, i, rand_idx):
        feature = np.zeros(self.feature_size, dtype=np.float)
        current_value_range = self.current_worst_value - self.current_best_value
        rand_X = self.X[rand_idx, :]
        rand_F = self.F[rand_idx]
        best_rand_value = np.min(rand_F)
        best_rand_solution = rand_X[np.argmin(rand_F)]

        feature[0] = (self.F[i]-self.current_best_value)/current_value_range
        feature[1] = (np.average(self.F)-self.current_best_value)/current_value_range
        feature[2] = np.std(self.F)/self.max_std
        feature[3] = (self.max_fe-self.n_fe)/self.max_fe
        feature[4] = self.dim/30
        feature[5] = self.stag_count/self.max_fe
        feature[6] = get_distance(self.X[i], self.X[rand_idx[0]])/self.max_dist
        feature[7] = get_distance(self.X[i], self.X[rand_idx[1]])/self.max_dist
        feature[8] = get_distance(self.X[i], self.X[rand_idx[2]])/self.max_dist
        feature[9] = get_distance(self.X[i], self.X[rand_idx[3]])/self.max_dist
        feature[10] = get_distance(self.X[i], self.X[rand_idx[4]])/self.max_dist
        feature[11] = get_distance(self.X[i], best_rand_solution)/self.max_dist
        feature[12] = (self.F[i]-self.F[rand_idx[0]])/current_value_range
        feature[13] = (self.F[i]-self.F[rand_idx[1]])/current_value_range
        feature[14] = (self.F[i]-self.F[rand_idx[2]])/current_value_range
        feature[15] = (self.F[i]-self.F[rand_idx[3]])/current_value_range
        feature[16] = (self.F[i]-self.F[rand_idx[4]])/current_value_range
        feature[17] = (self.F[i]-best_rand_value)/current_value_range
        feature[18] = get_distance(self.X[i], self.X[self.current_best_idx])/self.max_dist

        return feature

    def step(self, action):
        opr, scale = action[0], action[1]
        scale = scale[opr][0]
        mutate = mutation_operators[opr]
        x_prime = mutate(self.X, self.random_idx, scale, self.current_best_idx, self.current_idx)
        max_bound_violated_idx = x_prime > self.max_bounds
        min_bound_violated_idx = x_prime < self.min_bounds
        x_prime[max_bound_violated_idx] = self.max_bounds[max_bound_violated_idx]
        x_prime[min_bound_violated_idx] = self.min_bounds[min_bound_violated_idx]
        x_prime_value = self.func(x_prime)
        self.n_fe += 1
        reward = self.get_reward(self.current_idx, x_prime_value)

        if x_prime_value < self.F[self.current_idx]:
            if x_prime_value < self.current_best_value:
                self.current_best_value = x_prime_value
                self.current_best_idx = self.current_idx
            self.X[self.current_idx, :] = x_prime
            self.F[self.current_idx] = x_prime_value
        else:
            if x_prime_value > self.current_worst_value:
                self.current_worst_value = x_prime_value

        self.current_idx = (self.current_idx + 1) % self.NP
        self.random_idx = np.random.choice(self.NP, size=5, replace=False)
        feature = self.get_feature(self.current_idx,self.random_idx)
        is_terminal = self.n_fe >= self.max_fe

        # feature, reward, is_terminal flag, info
        print(self.n_fe, opr, scale, self.real_best_value, self.current_best_value, reward)
        return (feature, None), reward, is_terminal, None

    def get_reward(self, i, new_value):
        if self.reward_strategy == "R1":  # reward depends on performance
            reward = (self.F[i] - new_value)/(new_value-self.real_best_value)
        elif self.reward_strategy == "R2":    # with penalty
            reward = max((self.F[i] - new_value)/(new_value-self.real_best_value), 0)
        else:
            if self.reward_strategy == "R3":  # constant reward
                reward = 0
            else:   # incorporate penalty reward (R4)
                reward = -1
            if new_value < self.F[i]:
                reward = 1
                if new_value < self.current_best_value:
                    reward = 10
        return reward
