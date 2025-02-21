import pygame

pygame.init()


# game window
screen_width = 800
screen_height = 400

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('The Mürren games')

run = True
while run:


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

pygame.quit()
