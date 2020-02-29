from typing import NamedTuple, List, Iterator, Tuple, Union, Callable, Iterable

import torch
import torch.utils.data


class Experience(NamedTuple):
    """
    Represents one experience tuple for the Agent.
    """
    state: torch.FloatTensor
    action: int
    reward: float
    is_done: bool


class Episode(object):
    """
    Represents an entire sequence of experiences until a terminal state was
    reached.
    """

    def __init__(self, total_reward: float, experiences: List[Experience]):
        self.total_reward = total_reward
        self.experiences = experiences

    def calc_qvals(self, gamma: float) -> List[float]:
        """
        Calculates the q-value q(s,a), i.e. total discounted reward, for each
        step s and action a of a trajectory.
        :param gamma: discount factor.
        :return: A list of q-values, the same length as the number of
        experiences in this Experience.
        """
        qvals = []

        # TODO:
        #  Calculate the q(s,a) value of each state in the episode.
        #  Try to implement it in O(n) runtime, where n is the number of
        #  states. Hint: change the order.
        # ====== YOUR CODE: ======

        for cntr,exp in enumerate(reversed(self.experiences),0):
            if cntr == 0:
                
                qvals.append(exp.reward)
            else:
                qvals.append(exp.reward+gamma*qvals[-1])
        
        qvals.reverse()
        # ========================
        return qvals

    def __repr__(self):
        return f'Episode(total_reward={self.total_reward:.2f}, ' \
               f'#experences={len(self.experiences)})'


class TrainBatch(object):
    """
    Holds a batch of data to train on.
    """

    def __init__(self, states: torch.FloatTensor, actions: torch.LongTensor,
                 q_vals: torch.FloatTensor, total_rewards: torch.FloatTensor):

        assert states.shape[0] == actions.shape[0] == q_vals.shape[0]

        self.states = states
        self.actions = actions
        self.q_vals = q_vals
        self.total_rewards = total_rewards

    def __iter__(self):
        return iter(
            [self.states, self.actions, self.q_vals, self.total_rewards]
        )

    @classmethod
    def from_episodes(cls, episodes: Iterable[Episode], gamma=0.999):
        """
        Constructs a TrainBatch from a list of Episodes by extracting all
        experiences from all episodes.
        :param episodes: List of episodes to create the TrainBatch from.
        :param gamma: Discount factor for q-vals calculation
        """
        
        train_batch = None
        qvals_list = []
        states_list = []
        
        reward_list = []
        action_list = []
        
        # TODO:
        #   - Extract states, actions and total rewards from episodes.
        #   - Calculate the q-values for states in each experience.
        #   - Construct a TrainBatch instance.
        # ====== YOUR CODE: ======
        for episode in episodes :
            
            # concat lists of qvals 
            qvals_list += episode.calc_qvals(gamma)
            # extract states and actions
            for exp in episode.experiences:
                states_list.append(exp.state)
                action_list.append(exp.action)
            
            reward_list.append(episode.total_reward)
        qvals_tensor = torch.FloatTensor(qvals_list)
        qvals_tensor = torch.unsqueeze(qvals_tensor,1)

        states_tensor = torch.stack(states_list,0)

        
        rewards_tensor = torch.FloatTensor(reward_list)
        rewards_tensor = torch.unsqueeze(rewards_tensor,1)
        action_tensor = torch.LongTensor(action_list)
        action_tensor = torch.unsqueeze(action_tensor,1)
        
        train_batch = TrainBatch(states_tensor,action_tensor,qvals_tensor,rewards_tensor)

        # ========================
        return train_batch

    @property
    def num_episodes(self):
        return torch.numel(self.total_rewards)

    def __repr__(self):
        return f'TrainBatch(states: {self.states.shape}, ' \
               f'actions: {self.actions.shape}, ' \
               f'q_vals: {self.q_vals.shape}), ' \
               f'num_episodes: {self.num_episodes})'

    def __len__(self):
        return self.states.shape[0]


class TrainBatchDataset(torch.utils.data.IterableDataset):
    """
    This class generates batches of data for training a policy-based algorithm.
    It generates full episodes, in order for it to be possible to
    calculate q-values, so it's not very efficient.
    """

    def __init__(self, agent_fn: Callable, episode_batch_size: int,
                 gamma: float):
        """
        :param agent_fn: A function which accepts no arguments and returns
        an initialized agent ready to play.
        :param episode_batch_size: Number of episodes in each returned batch.
        :param gamma: discount factor for q-value calculation.
        """
        self.agent_fn = agent_fn
        self.gamma = gamma
        self.episode_batch_size = episode_batch_size

    def episode_batch_generator(self) -> Iterator[Tuple[Episode]]:
        """
        A generator function which (lazily) generates batches of Episodes
        from the Experiences of an agent.
        :return: A generator, each element of which will be a tuple of length
        batch_size, containing Episode objects.
        """
        curr_batch = []
        episode_reward = 0.0
        episode_experiences = []

        agent = self.agent_fn()
        agent.reset()

        while True:
            # TODO:
            #  - Play the environment with the agent until an episode ends.
            #  - Construct an Episode object based on the experiences generated
            #    by the agent.
            #  - Store Episodes in the curr_batch list.
            # ====== YOUR CODE: ======
            
            # Play for an entire episode  
            episode_done = False 
            while not episode_done:
                exp = agent.step()
                _,_,reward,episode_done = exp
                episode_reward += reward
                episode_experiences.append(exp)
            
            curr_batch.append(Episode(episode_reward,episode_experiences)) 
             
            # Initalise episode reward and episode_experiences for next episode iteration 
            episode_reward = 0.0
            episode_experiences = []

            # ========================
            if len(curr_batch) == self.episode_batch_size:
                yield tuple(curr_batch)
                curr_batch = []

    def __iter__(self) -> Iterator[TrainBatch]:
        """
        Lazily creates training batches from batches of Episodes.
        :return: A generator over instances of TrainBatch.
        """
        for episodes in self.episode_batch_generator():
            yield TrainBatch.from_episodes(episodes, self.gamma)
