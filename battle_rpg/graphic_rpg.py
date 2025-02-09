import pygame

pygame.init()

# pygame window
screen_width = 800
screen_heigth = 400

screen = pygame.display.set_mode((screen_width, screen_heigth))
pygame.display.set_caption('Murren final boss battle')

# load images
# Game background
background_image=pygame.image.load('img/Background/background.jpg').convert_alpha()

# function for drawing image
def draw_bg():
    screen.blit(background_image, (0,0))

run = True
while run:

    # draw background
    draw_bg()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    pygame.display.update()

pygame.quit()