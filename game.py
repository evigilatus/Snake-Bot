import pygame
from player import Player
from food import Food

# Set options to activate or deactivate the game view, and its speed
display_option = False
speed = 0
pygame.font.init()


class Game:

    def __init__(self, game_width, game_height):
        pygame.display.set_caption('SnakeGen')
        self.game_width = game_width
        self.game_height = game_height
        # self.gameDisplay = pygame.display.set_mode((game_width, game_height+60))
        #self.bg = pygame.image.load("img/background.png")
        self.crash = False
        self.player = Player(self)
        self.food = Food()
        self.score = 0
