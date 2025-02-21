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
background_img = pygame.image.load('img/Background/main_bg_resized.jpg').convert_alpha()

# Buttons
ch_1_img = pygame.image.load('img/Icons/RPG_Button_1_noBg copy.png').convert_alpha()
ch_2_img = pygame.image.load('img/Icons/RPG_Button_2_noBg copy.png').convert_alpha()

# create function for drawing text 
def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    screen.blit(img,(x, y))


# Function for drawing background
def draw_bg():
    screen.blit(background_img, (0,0))

# create buttons
chapter_1_button = button.Button(screen, 150, 100, ch_1_img, 500, 250)
chapter_2_button = button.Button(screen, 150, 200, ch_2_img, 500, 250)

# start a new chapter of the game
def start_chapter_2():
    # import chapter module
    subprocess.run([sys.executable, 'graphic_rpg.py'])
    sys.exit()

run = True
while run:
    clock.tick(fps)

    # Draw background
    draw_bg()

    # draw title text
    draw_text("The Mürren games", font, blue, 220, 50)

    # draw button and check click 
    pos = pygame.mouse.get_pos()
    
    # Check and draw buttons
    if chapter_1_button.draw():
        print('Chapter 1 button pressed')
    
    if chapter_2_button.draw():
        print('Chapter 2 button pressed')
        subprocess.Popen([sys.executable, 'graphic_rpg.py'])
        #subprocess.run([sys.executable, 'graphic_rpg.py'])
        #sys.exit()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
    
    pygame.display.update()

pygame.quit()
