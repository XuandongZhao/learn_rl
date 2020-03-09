import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class MAB(ABC):
    """
    Abstract class that represents a multi-armed bandit (MAB)
    """

    @abstractmethod
    def play(self, tround, context):
        """
        Play a round

        Arguments
        =========
        tround : int
            positive integer identifying the round

        context : 1D float array, shape (self.ndims * self.narms), optional
            context given to the arms

        Returns
        =======
        arm : int
            the positive integer arm id for this round
        """

    @abstractmethod
    def update(self, arm, reward, context):
        """
        Updates the internal state of the MAB after a play

        Arguments
        =========
        arm : int
            a positive integer arm id in {1, ..., self.narms}

        reward : float
            reward received from arm

        context : 1D float array, shape (self.ndims * self.narms), optional
            context given to arms
        """


class EpsGreedy(MAB):
    """
    Epsilon-Greedy multi-armed bandit

    Arguments
    =========
    narms : int
        number of arms

    epsilon : float
        explore probability

    Q0 : float, optional
        initial value for the arms
    """

    def __init__(self, narms, epsilon, Q0=np.inf):
        assert (narms > 0), "narms must be positive integers"
        assert (type(epsilon) == float), "epsilon must be floating number"

        self.narms = narms
        self.epsilon = epsilon
        self.Q0 = Q0
        self.arm_visit_count = {}
        self.arm_total_reward = {}
        self.arm_with_Q = {}

        for arm in range(1, self.narms + 1):
            self.arm_with_Q[arm] = self.Q0  # Initial all the arm with Q0
            self.arm_visit_count[arm] = 0  # Initial all the arm with zero number of visits
            self.arm_total_reward[arm] = 0  # Initial all the arm with zero reward

    def play(self, tround, context=None):
        assert (tround > 0), "tround must be positive integers"
        if np.random.random() < self.epsilon:  # exploration(Random select an arm)
            action = np.random.choice(self.narms)
        else:  # Select the arm with highest Q value
            # Getting the highest value from Q, then find the corresponding key and append them
            highest = max(self.arm_with_Q.values())
            highest_Qs = [key for key, value in self.arm_with_Q.items() if value == highest]
            if len(highest_Qs) > 1:
                action = np.random.choice(highest_Qs)  # Tie Breaker
            else:
                action = highest_Qs[0]
        return action

    def update(self, arm, reward, context=None):
        assert (arm > 0 and arm <= self.narms), "arm must be positive integers and no larger than self.narms"
        assert (type(reward) == float or type(reward) == np.float64), "reward must be floating point"
        self.arm_visit_count[arm] += 1
        self.arm_total_reward[arm] += reward
        updated_reward = self.arm_total_reward[arm] / self.arm_visit_count[arm]  # From lecture 13 slide pg7
        self.arm_with_Q.update({arm: updated_reward})

        return self.arm_with_Q


class UCB(MAB):
    """
    Upper Confidence Bound (UCB) multi-armed bandit

    Arguments
    =========
    narms : int
        number of arms

    rho : float
        positive real explore-exploit parameter

    Q0 : float, optional
        initial value for the arms
    """

    def __init__(self, narms, rho, Q0=np.inf):
        assert (narms > 0), "narms must be positive integers"
        assert (type(rho) == float and rho > 0.0 and np.isreal(rho)), "rho must be positive real floating number"

        self.narms = narms
        self.rho = rho
        self.Q0 = Q0
        self.arm_visit_count = {}
        self.arm_total_reward = {}
        self.arm_with_avg_reward = {}

        for arm in range(1, self.narms + 1):
            self.arm_with_avg_reward[arm] = self.Q0  # Initial all the arm with Q0

            self.arm_visit_count[arm] = 0  # Initial all the arm with zero number of visits
            self.arm_total_reward[arm] = 0  # Initial all the arm with zero reward

    def play(self, tround, context=None):
        assert (tround > 0), "tround must be positive integers"
        # copy each arm with reward for later calculation at tround
        temp_arm_with_Q = self.arm_with_avg_reward

        for arm in temp_arm_with_Q:
            if self.arm_visit_count[arm] == 0:  # Use Q0 for the first round
                continue
            else:
                # At tround, calculate Q with exlpore boost for each arm
                explore_boost_const = self.rho * np.log(tround) / self.arm_visit_count[arm]
                temp_arm_with_Q[arm] = temp_arm_with_Q[arm] + np.sqrt(explore_boost_const)

        # Getting the highest value from Q, then find the corresponding key and append them
        highest = max(temp_arm_with_Q.values())
        highest_Qs = [key for key, value in temp_arm_with_Q.items() if value == highest]
        if len(highest_Qs) > 1:
            action = np.random.choice(highest_Qs)  # Tie Breaker
        else:
            action = highest_Qs[0]
        return action

    def update(self, arm, reward, context=None):
        assert (arm > 0 and arm <= self.narms), "arm must be positive integers and no larger than self.narms"
        assert (type(reward) == float or type(reward) == np.float64), "reward must be floating point"
        # same as e-greedy
        self.arm_visit_count[arm] += 1
        self.arm_total_reward[arm] += reward
        updated_reward = self.arm_total_reward[arm] / self.arm_visit_count[arm]

        self.arm_with_avg_reward.update({arm: updated_reward})

        return self.arm_with_avg_reward


class BetaThompson(MAB):
    """
    Beta-Bernoulli Thompson sampling multi-armed bandit

    Arguments
    =========
    narms : int
        number of arms

    alpha0 : float, optional
        positive real prior hyperparameter

    beta0 : float, optional
        positive real prior hyperparameter
    """

    def __init__(self, narms, alpha0=1.0, beta0=1.0):
        assert (narms > 0), "narms must be positive integers"
        assert (type(alpha0) == float and alpha0 > 0.0 and np.isreal(
            alpha0)), "alpha0 must be real positive floating number"
        assert (type(beta0) == float and beta0 > 0.0 and np.isreal(
            beta0)), "beta0 must be real positive floating number"

        self.narms = narms
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.theta = np.random.beta(self.alpha0, self.beta0)  # Initial theta based on default alpha and beta

        self.arm_with_theta = {}
        self.arm_with_alpha0 = {}
        self.arm_with_beta0 = {}

        for arm in range(1, self.narms + 1):
            self.arm_with_theta[arm] = self.theta  # Initial all the arm with theta
            self.arm_with_alpha0[arm] = self.alpha0  # Initial all the arm with default alpha0
            self.arm_with_beta0[arm] = self.beta0  # Initial all the arm with default beta0

    def play(self, tround, context=None):
        assert (tround > 0), "tround must be positive integers"
        for arm in range(1, self.narms + 1):
            # For each arm, calculate theta
            self.arm_with_theta[arm] = np.random.beta(self.arm_with_alpha0[arm] + 1, self.arm_with_beta0[arm] + 1)
        # Getting the highest value from theta, then find the corresponding key and append them
        highest = max(self.arm_with_theta.values())
        highest_Qs = [key for key, value in self.arm_with_theta.items() if value == highest]
        if len(highest_Qs) > 1:
            action = np.random.choice(highest_Qs)  # Tie Breaker
        else:
            action = highest_Qs[0]
        return action

    def update(self, arm, reward, context=None):
        assert (arm > 0 and arm <= self.narms), "arm must be positive integers and no larger than self.narms"
        assert (type(reward) == float or type(reward) == np.float64), "reward must be floating point"
        # update alpha if there is a reward, in other case update beta.
        if reward == 1:
            self.arm_with_alpha0[arm] += 1
        else:
            self.arm_with_beta0[arm] += 1
