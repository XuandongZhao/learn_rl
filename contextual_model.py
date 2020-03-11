import numpy as np
from model import MAB


class LinUCB(MAB):
    """
    Contextual multi-armed bandit (LinUCB)

    Arguments
    =========
    narms : int
        number of arms

    ndims : int
        number of dimensions for each arm's context

    alpha : float
        positive real explore-exploit parameter
    """

    def __init__(self, narms, ndims, alpha):
        assert (narms > 0), "narms must be positive integers"
        assert (ndims > 0), "ndims must be positive integers"
        assert ((type(alpha) == float or type(alpha) == np.float64) and alpha > 0.0 and np.isreal(
            alpha)), "alpha must be real positive floating number"
        self.narms = narms
        self.ndims = ndims
        self.alpha = alpha

        self.A_a = {}  # A is the list of each arm with D.T * D + I
        self.b_a = {}  # b is the reward list

        for arm in range(1, self.narms + 1):
            if arm not in self.A_a:  # If arm is new
                # For each arm, initial identity matrix with feature dimensitonal space
                self.A_a[arm] = np.identity(self.ndims)
            if arm not in self.b_a:
                # For each arm, initial reward matrix with zeros. Dimension is each corresponding context's dimension
                self.b_a[arm] = np.zeros(self.ndims)

    def play(self, tround, context):
        assert (tround > 0), "tround must be positive integers"
        assert (context.shape == (
            self.narms * self.ndims,)), "context must be a numeric array of length self.ndims * self.narms"
        arm_with_Q = {}  # At tround, initial arm with empty posterior distribution

        context = context.reshape(self.narms, self.ndims)

        for arm in range(1, self.narms + 1):
            # For each arm, calculate posterior distribution based on theta and std
            Theta_a = np.dot(np.linalg.inv(self.A_a[arm]), self.b_a[arm])
            std = np.sqrt(
                np.linalg.multi_dot([np.transpose(context[arm - 1]), np.linalg.inv(self.A_a[arm]), context[arm - 1]]))
            p_ta = np.dot(Theta_a.T, context[arm - 1]) + self.alpha * std

            if not np.isnan(p_ta):  # make sure the result of calculation is valid number
                arm_with_Q[arm] = p_ta

        # Getting the highest value from posterior distribution, then find the corresponding key and append them
        highest = max(arm_with_Q.values())
        highest_Qs = [key for key, value in arm_with_Q.items() if value == highest]
        if len(highest_Qs) > 1:
            action = np.random.choice(highest_Qs)  # Tie Breaker
        else:
            action = highest_Qs[0]

        return action

    def update(self, arm, reward, context):
        assert (arm > 0 and arm <= self.narms), "arm must be positive integers and no larger than self.narms"
        assert (type(reward) == float or type(reward) == np.float64), "reward must be floating point"
        assert (context.shape == (
            self.narms * self.ndims,)), "context must be a numeric array of length self.ndims * self.narms"

        context = context.reshape(self.narms, self.ndims)

        if arm <= self.narms:
            # Reshap the vector to matrix, or the calculation will be incorrect
            # because the transpose will not take effects
            context_Matrix = context[arm - 1].reshape(-1, 1)
            context_times_contextT = np.dot(context_Matrix, context_Matrix.T)

            self.A_a[arm] = np.add(self.A_a[arm], context_times_contextT)

            self.b_a[arm] = np.add(self.b_a[arm], np.dot(reward, context[arm - 1]))


class LinThompson(MAB):
    """
    Contextual Thompson sampled multi-armed bandit (LinThompson)

    Arguments
    =========
    narms : int
        number of arms

    ndims : int
        number of dimensions for each arm's context

    v : float
        positive real explore-exploit parameter
    """

    def __init__(self, narms, ndims, v):
        assert (narms > 0), "narms must be positive integers"
        assert (ndims > 0), "ndims must be positive integers"
        # assert (type(v) == float and v > 0.0 and np.isreal(v)), "v must be real positive floating number"
        self.narms = narms
        self.ndims = ndims
        self.v = v
        self.B = np.identity(self.ndims)  # Initial B with identity matrix which has ndims dimension
        self.f = np.zeros(self.ndims)  # Initial total payoff with ndims of zeros
        self.u = np.zeros(self.ndims)  # Initial parameter mu with ndims of zeros

    def play(self, tround, context):
        assert (tround > 0), "tround must be positive integers"
        assert (context.shape == (
            self.narms * self.ndims,)), "context must be a numeric array of length self.ndims * self.narms"

        arm_with_Q = {}

        context = context.reshape(self.narms, self.ndims)
        # Calculate prior from multivariate Gaussian distribution
        u_t = np.random.multivariate_normal(self.u, self.v * self.v * np.linalg.inv(self.B))

        for arm in range(1, self.narms + 1):
            # calculate posterior distribution for each arm
            arm_with_Q[arm] = np.dot(np.transpose(context[arm - 1]), u_t)

        # Getting the highest value from posterior distribution, then find the corresponding key and append them
        highest = max(arm_with_Q.values())
        highest_Qs = [key for key, value in arm_with_Q.items() if value == highest]
        if len(highest_Qs) > 1:
            action = np.random.choice(highest_Qs)  # Tie Breaker
        else:
            action = highest_Qs[0]
        return action

    def update(self, arm, reward, context):
        assert (arm > 0 and arm <= self.narms), "arm must be positive integers and no larger than self.narms"
        assert (type(reward) == float or type(reward) == np.float64), "reward must be floating point"
        assert (context.shape == (
            self.narms * self.ndims,)), "context must be a numeric array of length self.ndims * self.narms"

        context = context.reshape(self.narms, self.ndims)

        if arm <= self.narms:
            # Reshap the vector to matrix, or the calculation will be incorrect
            # because the transpose will not take effects
            context_Matrix = context[arm - 1].reshape(-1, 1)
            context_times_contextT = np.dot(context_Matrix, context_Matrix.T)
            # Update B
            self.B = np.add(self.B, context_times_contextT)
            # Update reward f
            self.f = np.add(self.f, np.multiply(reward, context[arm - 1]))
            # Update mu
            self.u = np.dot(np.linalg.inv(self.B), self.f)
