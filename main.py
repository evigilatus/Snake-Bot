import click

import pygame
from game import Game
from agents.dueling_dqn import DuelingDQNAgent
from agents.experience_replay_dqn import ExperienceReplayDQNAgent
from agents.double_dqn_periodic import DoubleDQNPeriodicAgent
from utils.settings import game_settings

SETTINGS = './game_modes/settings.py'

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
@click.option('--graphics/--no-graphics', default=True, help='Show UI')
@click.option('--weights-path', required=False, type=click.Path(),
              help='Pass the path to a \'*.hdf5\' file with the network weights from a previous training '
                   'to use a pretrained model. '
                   '\nNOTE: The path must lead to a file corresponding to the selected agent type!' )
def main(algorithm, mode, graphics, weights_path):
    init_settings(graphics, weights_path)
    agent = select_algorithm(algorithm)
    mode_file = select_mode(mode)

    print(f"Running {algorithm} agent for solving the game 'Snake' in {mode} mode...")
    agent.run(mode_file)


def init_settings(graphics, weights_path):
    game_settings['weights_path'] = weights_path
    game_settings['display_option'] = graphics
    game_settings['speed'] = 0 if graphics else None
    if graphics:
        pygame.font.init()


def select_algorithm(algorithm):
    if algorithm == 'experience_replay_dqn':
        agent = ExperienceReplayDQNAgent()
    elif algorithm == 'double_dqn':
        agent = DoubleDQNPeriodicAgent()
    elif algorithm == 'dueling_dqn':
        agent = DuelingDQNAgent()
    else:
        agent = DuelingDQNAgent()
    return agent


def select_mode(mode):
    mode_file = ''
    if mode == 'wall':
        mode_file = 'game_modes/wall.json'
    elif mode == 'maze':
        mode_file = 'game_modes/maze.json'
    return mode_file


if __name__ == '__main__':
    main()


