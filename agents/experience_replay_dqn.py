from utils.memory import Memory
from random import randint
from agents.dqn_agent import DQNAgent
from keras.utils import to_categorical
from game import Game
from game_initializer import *
from utils.settings import game_settings


class ExperienceReplayDQNAgent(DQNAgent):

    def __init__(self):
        super().__init__()
        self.memory = Memory(1000)

    def remember(self, state, action, reward, next_state, done):
        self.memory.store((state, action, reward, next_state, done))

    def replay_new(self, memory):
        idx, minibatch, ISWeights = memory.sample(1000)
        for sample in minibatch:
            state = sample[0][0]
            action = sample[0][1]
            reward = sample[0][2]
            next_state = sample[0][3]
            done = sample[0][4]
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            memory.batch_update(idx, np.abs(target_f[0] - target))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, 11)))[0])
        target_f = self.model.predict(state.reshape((1, 11)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, 11)), target_f, epochs=1, verbose=0)

    def run(self, mode_file):
        pygame.init()
        counter_games = 0
        score_plot = []
        counter_plot = []
        record = 0
        while counter_games < 200:
            # Initialize classes
            game = Game(440, 440, mode_file)
            player1 = game.player
            food1 = game.food

            # Perform first move
            initialize_game(player1, game, food1, self)
            if game_settings['display_option']:
                display(player1, food1, game, record)

            while not game.crash:
                # agent.epsilon is set to give randomness to actions
                self.epsilon = 80 - counter_games

                # get old state
                state_old = self.get_state(game, player1, food1)

                # perform random actions based on agent.epsilon, or choose the action
                if randint(0, 200) < self.epsilon:
                    final_move = to_categorical(randint(0, 2), num_classes=3)
                else:
                    # predict action based on the old state
                    prediction = self.model.predict(state_old.reshape((1, 11)))
                    final_move = to_categorical(np.argmax(prediction[0]), num_classes=3)

                # perform new move and get new state
                player1.do_move(final_move, player1.x, player1.y, game, food1, self)
                state_new = self.get_state(game, player1, food1)

                # set reward for the new state
                reward = self.set_reward(player1, game.crash)

                # train short memory base on the new action and state
                self.train_short_memory(state_old, final_move, reward, state_new, game.crash)

                # store the new data into a long term memory
                self.remember(state_old, final_move, reward, state_new, game.crash)
                record = get_record(game.score, record)
                if game_settings['display_option']:
                    display(player1, food1, game, record)
                    pygame.time.wait(game_settings['speed'])

            self.replay_new(self.memory)
            counter_games += 1
            print('Game', counter_games, '      Score:', game.score)
            score_plot.append(game.score)
            counter_plot.append(counter_games)
        self.model.save_weights('weights.hdf5')
        # from google.colab import files
        # files.download("weights.hdf5")
        plot_seaborn(counter_plot, score_plot)