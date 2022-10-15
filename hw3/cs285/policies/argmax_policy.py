import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        ## TODO-Done return the action that maxinmizes the Q-value 
        # at the current observation as the output

        q_vals = self.critic.qa_values(observation)
        action = np.argmax(q_vals, axis=1)

        return action.squeeze()