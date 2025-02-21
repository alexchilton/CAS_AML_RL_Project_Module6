import pygame
import button
import subprocess
import sys

pygame.init()

clock = pygame.time.Clock()
fps = 60

# game window
screen_width = 800
screen_height = 400

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('The Mürren games')

clicked = False

# Define fonts
font = pygame.font.SysFont('copperplategothic', 35)

# define colors
blue= (0, 0, 255)
powderblue= (176,224,230)

# load images 
# background image
background_img = pygame.image.load('img/Background/ch2_bg_resized.jpg').convert_alpha()

# Buttons
player_img = pygame.image.load('img/Icons/RPG_Button_player.png').convert_alpha()
agent_img = pygame.image.load('img/Icons/RPG_Button_agent.png').convert_alpha()

# create function for drawing text 
def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    screen.blit(img,(x, y))


# Function for drawing background
def draw_bg():
    screen.blit(background_img, (0,0))

# create buttons
player_button = button.Button(screen, 190, 170, player_img, 165, 62)
agent_button = button.Button(screen, 450, 170, agent_img, 165, 62)

run = True
while run:
    clock.tick(fps)

    # Draw background
    draw_bg()

    # draw title text
    draw_text("The battle challenge", font, blue, 190, 50)

    # draw button and check click 
    pos = pygame.mouse.get_pos()

    # Check and draw buttons
    if player_button.draw():
        subprocess.Popen([sys.executable, 'graphic_rpg.py'])
    
    if agent_button.draw():
        subprocess.Popen([sys.executable, 'graphic_demo.py'])
        #subprocess.run([sys.executable, 'graphic_rpg.py'])
        #sys.exit()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    
    pygame.display.update()

pygame.quit()


