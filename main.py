import click

import pygame
from random import randint
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# from google.colab import files
from agents.dueling_dqn import DuelingDQNAgent
from agents.experience_replay_dqn import ExperienceReplayDQNAgent
from game import Game

@click.group(invoke_without_command=True)
@click.option('--algorithm', required=False,
              help='Select RL algorithm from the following: '
                   '\n- \'experience_replay_dqn\' (Prioritized Experience Replay DQN)'
                   '\n- \'double_dqn (Double DQN)\''
                   '\n- \'dueling_dqn (Dueling DQN)\'')
@click.option('--mode', required=False, default='standard',
              help='Select game-mode from the following: '
                   '\n- \'standard\''
                   '\n- \'wall\''
                   '\n- \'maze\'')
def main(algorithm, mode):
    # TODO: Add wall/maze logic
    if algorithm == 'experience_replay_dqn':
        print("Running experiments with ExperienceReplayDQNAgent")
        agent = ExperienceReplayDQNAgent()
    elif algorithm == 'double_dqn':
        print("Running experiments with DoubleDQN")
        # TODO: Fix it with the correct double DQN agent!
        agent = DuelingDQNAgent()
    elif algorithm == 'dueling_dqn':
        print("Running experiments with DuelingDQN")
        agent = DuelingDQNAgent()
    else:
        # TODO: Choose default agent
        agent = DuelingDQNAgent()

    agent.run()


if __name__ == '__main__':
    main()


