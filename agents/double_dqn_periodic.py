import random
from random import randint
from keras.utils import to_categorical
from agents.dqn_agent import DQNAgent
from game import Game
from game_initializer import *
from utils.settings import game_settings


class DoubleDQNPeriodicAgent(DQNAgent):

    def __init__(self):
        super().__init__()
        self.q_model = self.network()
        self.target_model = self.network()
        self.update_counter = 0  # only update target network after this counter hits certain value

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory):
        if len(memory) > 1000:
            minibatch = random.sample(memory, 1000)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                action_index = np.argmax(self.q_model.predict(next_state.reshape((1, 11)))[0])
                target = reward + self.gamma * self.target_model.predict(next_state.reshape((1, 11)))[0][action_index]
            # self.target_model = self.q_model
            target_f = self.q_model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.q_model.fit(np.array([state]), target_f, epochs=1, verbose=0)
            self.target_model.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            # Use target-network to predict action selection for next_state
            action_index = np.argmax(self.q_model.predict(next_state.reshape((1, 11)))[0])
            # Use q-network with the index of the above predicted action for action evaluation
            target = reward + self.gamma * self.target_model.predict(next_state.reshape((1, 11)))[0][action_index]
        # self.target_model = self.q_model
        target_f = self.q_model.predict(state.reshape((1, 11)))
        target_f[0][np.argmax(action)] = target
        self.q_model.fit(state.reshape((1, 11)), target_f, epochs=1, verbose=0)
        self.update_counter = self.update_counter + 1
        if self.update_counter % 5 == 0:
            self.target_model.fit(state.reshape((1, 11)), target_f, epochs=1, verbose=0)

    def run(self, mode_file):
        pygame.init()
        counter_games = 0
        score_plot = []
        counter_plot = []
        record = 0
        while counter_games < 150:
            # Initialize classes
            game = Game(440, 440, mode_file)
            player1 = game.player
            food1 = game.food

            # Perform first move
            game = Game(440, 440, mode_file)
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
                    prediction = self.q_model.predict(state_old.reshape((1, 11)))
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
        self.q_model.save_weights('wall_weights_q.hdf5')
        self.target_model.save_weights('wall_weights_target.hdf5')
        plot_seaborn(counter_plot, score_plot)
        # files.download("weights_q.hdf5")
        # files.download("weights_target.hdf5")
