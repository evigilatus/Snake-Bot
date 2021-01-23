from random import randint
import pygame

from game_initializer import update_screen
from utils.settings import game_settings


class Food(object):

    def __init__(self):
        self.x_food = 220
        self.y_food = 200
        if game_settings['display_option']:
            self.image = pygame.image.load('./game_modes/img/food2.png')

    def food_coord(self, game, player):
        x_rand = randint(20, game.game_width - 40)
        self.x_food = x_rand - x_rand % 20
        y_rand = randint(20, game.game_height - 40)
        self.y_food = y_rand - y_rand % 20
        if [self.x_food, self.y_food] not in player.position and [self.x_food, self.y_food] not in game.barrierPositions:
            return self.x_food, self.y_food
        else:
            self.food_coord(game,player)

    def display_food(self, x, y, game):
        game.gameDisplay.blit(self.image, (x, y))
        update_screen()






