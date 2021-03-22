import numpy as np
from optproblems.cec2005 import *
import torch
import sys

from agent import Agent
import de_test

max_cycle = 1000
#Env and Agent Params
# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')
num_inputs = 99
num_outputs = 4

def test_agent(agent, env):
    ob = env.reset()
    episodic_reward = 0
    end_flag = False
    runs = 1
    while not end_flag:
        # get action pair (operator, F), add 1 to F so F range (0, 2)
        op, op_onehot, op_prob, F, F_prob, value = agent.choose_action(ob)
        action = (op, F)
        ob, end_flag = env.step(action=action)
        print("ACTION (",op,",",F,")"," BUDGET", env.budget, ": ", env.copy_F[env.best])
        if end_flag and runs < 25:
            runs += 1
            ob = env.reset()
            end_flag = False

def test(argv):
    ENV_NAME = argv[1]

    d = [10, 30]
    func_select = [unimodal.F3, basic_multimodal.F9, f16.F16, f18.F18, f23.F23]
    for i in range(2):
        for j in range(5):
            dim = d[i]; #print(dim)
            fun = func_select[j](dim)
            lbounds = fun.min_bounds; lbounds = np.array(lbounds); #print(lbounds)
            ubounds = fun.max_bounds; ubounds = np.array(ubounds); #print(ubounds)
            opti = fun.get_optimal_solutions()
            for o in opti:
                print(o.phenome, o.objective_values)

            sol = np.copy(o.phenome)
            best_value = fun.objective_function(sol)

            agent = Agent(num_inputs=[num_inputs], num_outputs=num_outputs, device=DEVICE, env_name=ENV_NAME, training=False)
            print(" best value= ",best_value)
            #for repeat in range(10):
            env = de_test.DEEnv(fun, lbounds, ubounds, dim, best_value)
            test_agent(agent, env)

if __name__ == "__main__":
    test(sys.argv)