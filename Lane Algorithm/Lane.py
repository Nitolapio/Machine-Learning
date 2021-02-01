import pygame
import sphinx

class Lane:
    def __init__(self, width, height, color):
        self.width = width
        self.height = height
        self.color = color

    def __str__(self):
        return f'{self.width}, {self.height}, {self.color}'

    def getWidth(self):
        return self.width
    
    def getHeigth(self):
        return self.height  
    
    def getColor(self):
        return self.color

    def drawLane(self, screen, xPos, yPos):
        '''
        :param screen: screen to draw on
        :param int xPos: x Position
        :param int yPos: y Position
        '''
        pygame.draw.rect(screen, self.color, (xPos, yPos, self.width, self.height))