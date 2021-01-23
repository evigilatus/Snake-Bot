import click

import pygame
from game import Game
from agents.dueling_dqn import DuelingDQNAgent
from agents.experience_replay_dqn import ExperienceReplayDQNAgent
from agents.double_dqn_periodic import DoubleDQNPeriodicAgent
from utils.settings import game_settings


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
@click.option('--graphics', default=True, type=click.BOOL, help='Show UI')
def main(algorithm, mode, graphics):
    init_graphics(graphics)
    
    if algorithm == 'experience_replay_dqn':
        agent = ExperienceReplayDQNAgent()
    elif algorithm == 'double_dqn':
        agent = DoubleDQNPeriodicAgent()
    elif algorithm == 'dueling_dqn':
        agent = DuelingDQNAgent()
    else:
        # TODO: Choose default agent
        agent = DuelingDQNAgent()
    
    mode_file = ''
    
    if mode == 'wall':
        mode_file = 'game_modes/wall.json'
    elif mode == 'maze':
        mode_file = 'game_modes/maze.json'
        
    game = Game(440, 440, mode_file)

    agent.run(game)


def init_graphics(graphics):
    game_settings['display_option'] = graphics
    game_settings['speed'] = 0 if graphics else None
    if graphics:
        pygame.font.init()


if __name__ == '__main__':
    main()


