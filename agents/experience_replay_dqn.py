from agents.memory import Memory
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd
from operator import add
from random import randint
from keras.utils import to_categorical
from game import Game
from game_helper import *


class ExperienceReplayDQNAgent(object):

    def __init__(self):
        self.reward = 0
        self.gamma = 0.9
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = 0.0005
        self.model = self.network()
        # self.model = self.network("weights.hdf5")
        self.epsilon = 0
        self.actual = []
        self.memory = Memory(1000)

    def get_state(self, game, player, food):

        state = [
            (player.x_change == 20 and player.y_change == 0 and (
                        (list(map(add, player.position[-1], [20, 0])) in player.position) or
                        player.position[-1][0] + 20 >= (game.game_width - 20))) or (
                        player.x_change == -20 and player.y_change == 0 and (
                            (list(map(add, player.position[-1], [-20, 0])) in player.position) or
                            player.position[-1][0] - 20 < 20)) or (player.x_change == 0 and player.y_change == -20 and (
                        (list(map(add, player.position[-1], [0, -20])) in player.position) or
                        player.position[-1][-1] - 20 < 20)) or (player.x_change == 0 and player.y_change == 20 and (
                        (list(map(add, player.position[-1], [0, 20])) in player.position) or
                        player.position[-1][-1] + 20 >= (game.game_height - 20))),  # danger straight

            (player.x_change == 0 and player.y_change == -20 and (
                        (list(map(add, player.position[-1], [20, 0])) in player.position) or
                        player.position[-1][0] + 20 > (game.game_width - 20))) or (
                        player.x_change == 0 and player.y_change == 20 and ((list(map(add, player.position[-1],
                                                                                      [-20, 0])) in player.position) or
                                                                            player.position[-1][0] - 20 < 20)) or (
                        player.x_change == -20 and player.y_change == 0 and ((list(map(
                    add, player.position[-1], [0, -20])) in player.position) or player.position[-1][-1] - 20 < 20)) or (
                        player.x_change == 20 and player.y_change == 0 and (
                        (list(map(add, player.position[-1], [0, 20])) in player.position) or player.position[-1][
                    -1] + 20 >= (game.game_height - 20))),  # danger right

            (player.x_change == 0 and player.y_change == 20 and (
                        (list(map(add, player.position[-1], [20, 0])) in player.position) or
                        player.position[-1][0] + 20 > (game.game_width - 20))) or (
                        player.x_change == 0 and player.y_change == -20 and ((list(map(
                    add, player.position[-1], [-20, 0])) in player.position) or player.position[-1][0] - 20 < 20)) or (
                        player.x_change == 20 and player.y_change == 0 and (
                        (list(map(add, player.position[-1], [0, -20])) in player.position) or player.position[-1][
                    -1] - 20 < 20)) or (
                    player.x_change == -20 and player.y_change == 0 and (
                        (list(map(add, player.position[-1], [0, 20])) in player.position) or
                        player.position[-1][-1] + 20 >= (game.game_height - 20))),  # danger left

            player.x_change == -20,  # move left
            player.x_change == 20,  # move right
            player.y_change == -20,  # move up
            player.y_change == 20,  # move down
            food.x_food < player.x,  # food left
            food.x_food > player.x,  # food right
            food.y_food < player.y,  # food up
            food.y_food > player.y  # food down
        ]

        for i in range(len(state)):
            if state[i]:
                state[i] = 1
            else:
                state[i] = 0

        return np.asarray(state)

    def set_reward(self, player, crash):
        self.reward = 0
        if crash:
            self.reward = -10
            return self.reward
        if player.eaten:
            self.reward = 10
        return self.reward

    def network(self, weights=None):
        model = Sequential()
        model.add(Dense(120, activation='relu', input_dim=11))
        model.add(Dropout(0.15))
        model.add(Dense(120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(3, activation='softmax'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(weights)
        return model

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

    def run(self):
        pygame.init()
        counter_games = 0
        score_plot = []
        counter_plot = []
        record = 0
        while counter_games < 1000:
            # Initialize classes
            game = Game(440, 440)
            player1 = game.player
            food1 = game.food

            # Perform first move
            initialize_game(player1, game, food1, self)
            if display_option:
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

                # set treward for the new state
                reward = self.set_reward(player1, game.crash)

                # train short memory base on the new action and state
                self.train_short_memory(state_old, final_move, reward, state_new, game.crash)

                # store the new data into a long term memory
                self.remember(state_old, final_move, reward, state_new, game.crash)
                record = get_record(game.score, record)
                if display_option:
                    display(player1, food1, game, record)
                    pygame.time.wait(speed)

            self.replay_new(self.memory)
            counter_games += 1
            print('Game', counter_games, '      Score:', game.score)
            score_plot.append(game.score)
            counter_plot.append(counter_games)
        self.model.save_weights('weights.hdf5')
        # from google.colab import files
        # files.download("weights.hdf5")
        plot_seaborn(counter_plot, score_plot)