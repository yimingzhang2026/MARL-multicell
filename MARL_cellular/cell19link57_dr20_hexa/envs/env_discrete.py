import gym
from gym import spaces
import numpy as np
from envs.env_core import EnvCore


class DiscreteActionEnv(object):
    """
    Wrapper for discrete action environment.
    """

    def __init__(self,args):
        self.env = EnvCore(args)
        self.num_agent = self.env.agent_num
        self.max_num_neighbors = self.env.max_num_neighbors
        self.signal_obs_dim = self.env.obs_dim
        self.signal_action_dim_schedule = self.env.action_dim_schedule
        self.signal_action_dim_power = self.env.action_dim_power
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        share_obs_dim = [0] * self.num_agent
        total_action_space = []
        # self.action_space = [MultiDiscrete([[0, self.signal_action_dim_schedule - 1]] * self.env.M) for _ in range(self.num_agent)]
        self.action_space = [MultiDiscrete([[0, self.signal_action_dim_schedule - 1], [0, self.signal_action_dim_power - 1]] * self.env.M) for _ in range(self.num_agent)]
        for agent_idx in range(self.num_agent):
            # observation space
            share_obs_dim[agent_idx] = self.signal_obs_dim * (self.env.max_num_neighbors + 1)
            # share_obs_dim += self.signal_obs_dim
            # share_obs_dim = self.signal_obs_dim
            self.observation_space.append(
                spaces.Box(
                    low=-np.inf,
                    high=+np.inf,
                    shape=(self.signal_obs_dim,),
                    dtype=np.float32,
                )
            )  # [-inf,inf]

        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim[i],), dtype=np.float32)
            for i in range(self.num_agent)
        ]

    def step(self, actions):
        """
        """

        results = self.env.step(actions)
        obs, rews, dones, infos = results
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        obs = self.env.reset()
        return np.stack(obs)

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        pass

    def seed(self, seed):
        pass


class MultiDiscrete:
    """
    - The multi-discrete action space consists of a series of discrete action spaces with different parameters
    - It can be adapted to both a Discrete action space or a continuous (Box) action space
    - It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space
    - It is parametrized by passing an array of arrays containing [min, max] for each discrete action space
       where the discrete action space can take any integers from `min` to `max` (both inclusive)
    Note: A value of 0 always need to represent the NOOP action.
    e.g. Nintendo Game Controller
    - Can be conceptualized as 3 discrete action spaces:
        1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
    - Can be initialized as
        MultiDiscrete([ [0,4], [0,1], [0,1] ])
    """

    def __init__(self, array_of_param_array):
        super().__init__()
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]
        self.n = np.sum(self.high) + 2

    def sample(self):
        """Returns a array with one sample from each discrete action space"""
        # For each row: round(random .* (max - min) + min, 0)
        random_array = np.random.rand(self.num_discrete_space)
        return [int(x) for x in np.floor(np.multiply((self.high - self.low + 1.0), random_array) + self.low)]

    def contains(self, x):
        return (
            len(x) == self.num_discrete_space
            and (np.array(x) >= self.low).all()
            and (np.array(x) <= self.high).all()
        )

    @property
    def shape(self):
        return self.num_discrete_space

    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)

    def __eq__(self, other):
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)


if __name__ == "__main__":
    DiscreteActionEnv().step(actions=None)
