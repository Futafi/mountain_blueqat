import numpy as np
from blueqat import opt
import gym
Opt = opt.Opt
env = gym.make("MountainCar-v0")

# 運動のもつエネルギー的なのと位置のオーダーを合わせる補正
a = 10
# pushと運動のオーダーを合わせる補正
b = 5e-2
# 選択肢間の差が小さすぎるので差を広げる補正
c = 100


def get_matrix(position, velocity, a, b):
    # 中心を0に近づけ、負の値を正にする（位置エネルギー的に）
    e_p = abs(position + 0.5 )
    matrix = [-c*(e_p+a*abs(velocity + -b)), -c*(e_p+a * abs(velocity + b * 0)), -c*(e_p+a*abs(velocity + b))]
    return matrix

qstr = "1e2*(1 - (q0 + q1 + q2))^2"

# 001みたいなのを選択肢に直す
def get_action(action_qaoa):
    for i in range(len(action_qaoa)):
        if action_qaoa[i] == 1:
            return i

observation = env.reset()
actions = []
missed_count = 0
obs = []

for i in range(200):
    p,v = observation

    problem = Opt()
    # diag actionが一つだけ選ばれるように補正
    problem.add(np.diag(get_matrix(p, v, a, b))).add(qstr).add(np.diag([30,30,30]))
    if i == 0:
        print(problem.qubo)

    action_qaoa = problem.qaoa().most_common()[0][0]
    # actionが複数選ばれた時(111)繰り返し
    while sum(action_qaoa) != 1:
        missed_count +=1
        print("Failed")
        if missed_count >= 400:
            print("Failed")
        action_qaoa = problem.qaoa().most_common()[0][0]

    action = get_action(action_qaoa)

    actions.append(action)

    observation, reward, done, info = env.step(action)
    obs.append(observation)

    if done == True:
        break

if obs[-1][0] >= 0.5:
    print("Clear")
