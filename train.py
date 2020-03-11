import numpy as np
from model import EpsGreedy, UCB, BetaThompson
from contextual_model import LinThompson, LinUCB
from evaluate import offlineEvaluate
import matplotlib.pyplot as plt


def main():
    # Initial three lists which respects
    # to arm, reward, and context(feature_list in this case).
    arm_list = []
    reward_list = []
    features_list = []
    num_of_events = 0

    # Read each line and split by spaces. Record arm, reward and context into the lists
    with open("dataset.txt", "r") as f:
        dataset = f.readlines()
    for line in dataset:
        num_of_events += 1
        temp_line = line.split()
        arm = int(temp_line[0])
        reward = float(temp_line[1])
        features = temp_line[2:]
        features = list(map(float, features))
        arm_list.append(arm)
        reward_list.append(reward)
        features_list.append(features)

    # Convert lists into np_array
    arms = np.array(arm_list)
    rewards = np.array(reward_list)

    # For each event, the context is 10*10 dim
    # because there are 10 arms, and each one of them has 10 features
    contexts = np.array(features_list).reshape(num_of_events, (10 * 10))

    mab = EpsGreedy(10, 0.05)
    results_EpsGreedy = offlineEvaluate(mab, arms, rewards, contexts, 800)
    print(len(results_EpsGreedy))
    print('EpsGreedy average reward', np.mean(results_EpsGreedy))

    mab = UCB(10, 1.0)
    results_UCB = offlineEvaluate(mab, arms, rewards, contexts, 800)
    print('UCB average reward', np.mean(results_UCB))

    mab = BetaThompson(10, 1.0, 1.0)
    results_BetaThompson = offlineEvaluate(mab, arms, rewards, contexts, 800)
    print('BetaThompson average reward', np.mean(results_BetaThompson))

    mab = LinUCB(10, 10, 1.0)
    results_LinUCB = offlineEvaluate(mab, arms, rewards, contexts, 800)
    print('LinUCB average reward', np.mean(results_LinUCB))

    mab = LinThompson(10, 10, 1.0)
    results_LinThompson = offlineEvaluate(mab, arms, rewards, contexts, 800)
    print('LinThompson average reward', np.mean(results_LinThompson))

    EpsGreedy_reward = []
    UCB_reward = []
    betaThompson_reward = []
    linUCB_reward = []
    linThompson_reward = []
    total_round = []
    count = 0
    for round in range(1, 801):
        count += 1
        total_round.append(count)
        if round == 1:  # calculate initial reward
            EpsGreedy_reward.append(results_EpsGreedy[round - 1] / round)
            UCB_reward.append(results_UCB[round - 1] / round)
            betaThompson_reward.append(results_BetaThompson[round - 1] / round)
            linUCB_reward.append(results_LinUCB[round - 1] / round)
            linThompson_reward.append(results_LinThompson[round - 1] / round)
        else:  # calculate cumulated reward
            results_EpsGreedy[round - 1] += results_EpsGreedy[round - 2]
            EpsGreedy_reward.append(results_EpsGreedy[round - 1] / round)

            results_UCB[round - 1] += results_UCB[round - 2]
            UCB_reward.append(results_UCB[round - 1] / round)

            results_BetaThompson[round - 1] += results_BetaThompson[round - 2]
            betaThompson_reward.append(results_BetaThompson[round - 1] / round)

            results_LinUCB[round - 1] += results_LinUCB[round - 2]
            linUCB_reward.append(results_LinUCB[round - 1] / round)

            results_LinThompson[round - 1] += results_LinThompson[round - 2]
            linThompson_reward.append(results_LinThompson[round - 1] / round)

    plt.plot(total_round, EpsGreedy_reward, label="eGreedy")
    plt.plot(total_round, UCB_reward, label="UCB")
    plt.plot(total_round, betaThompson_reward, label="BetaThompson")
    plt.plot(total_round, linUCB_reward, label="LinUCB")
    plt.plot(total_round, linThompson_reward, label="LinThompson")
    plt.ylabel('Per-Round Cumulative Reward')
    plt.xlabel('Rounds')
    plt.legend()
    plt.savefig('compare.png')
    plt.show()


    # First Interval [0.2 0.4 0.6 0.8 1. ]
    alpha_range_one_decimal = np.linspace(0, 1, 6)
    alpha_range_one_decimal = np.delete(alpha_range_one_decimal, 0)  # delete zero
    # Second Interval [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 ]
    alpha_range_two_decimal = np.linspace(0, 0.1, 11)
    alpha_range_two_decimal = np.delete(alpha_range_two_decimal, 0)  # delete zero
    # Append two intervals and sort
    alpha_range = np.append(alpha_range_two_decimal, alpha_range_one_decimal)
    alpha_range = np.sort(alpha_range)
    results_LinUCB_with_alpha = []

    for alpha in alpha_range:
        mab = LinUCB(10, 10, alpha)
        results_LinUCB = offlineEvaluate(mab, arms, rewards, contexts, 800)
        results_LinUCB_with_alpha.append(np.mean(results_LinUCB))

    plt.plot(alpha_range, results_LinUCB_with_alpha, linestyle='dashed')
    plt.ylabel('mean_reward')
    plt.xlabel('apha_range')
    plt.savefig('LinUCB.png')
    plt.show()

    # First Interval [0.2 0.4 0.6 0.8 1. ]
    v_range_one_decimal = np.linspace(0, 1, 6)
    v_range_one_decimal = np.delete(v_range_one_decimal, 0)  # delete zero
    # Second Interval [0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 ]
    v_range_two_decimal = np.linspace(0, 0.1, 11)
    v_range_two_decimal = np.delete(v_range_two_decimal, 0)  # delete zero
    # Append two intervals and sort
    v_range = np.append(v_range_two_decimal, v_range_one_decimal)
    v_range = np.sort(v_range)

    resutls_LinThompson_with_v = []
    for v in v_range:
        mab = LinThompson(10, 10, v)
        results_LinThompson = offlineEvaluate(mab, arms, rewards, contexts, 800)
        resutls_LinThompson_with_v.append(np.mean(results_LinThompson))
    plt.plot(v_range, resutls_LinThompson_with_v, linestyle='dashed')
    plt.ylabel('mean_reward')
    plt.xlabel('v_range')
    plt.savefig('LinThom.png')
    plt.show()



if __name__ == '__main__':
    main()
