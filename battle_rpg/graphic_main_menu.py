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
small_font = pygame.font.SysFont('copperplategothic', 25)

# define colors
blue = (0, 0, 255)
powderblue = (176, 224, 230)
white = (255, 255, 255)
black = (0, 0, 0)

# load images 
# background image
background_img = pygame.image.load('img/Background/main_bg_resized.jpg').convert_alpha()

# Buttons
ch_1_img = pygame.image.load('img/Icons/RPG_Button_1_noBg copy.png').convert_alpha()
ch_2_img = pygame.image.load('img/Icons/RPG_Button_2_noBg copy.png').convert_alpha()
ch_3_img = pygame.image.load('img/Icons/RPG_Button_3_noBg copy.png').convert_alpha()
credits_img = pygame.image.load('img/Icons/credits.png').convert_alpha()
back_img = pygame.image.load('img/Icons/back.png').convert_alpha()

# create function for drawing text 
def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    screen.blit(img, (x, y))

# Function for drawing background
def draw_bg():
    screen.blit(background_img, (0, 0))

# create buttons
chapter_1_button = button.Button(screen, 190, 120, ch_1_img, 420, 65)
chapter_2_button = button.Button(screen, 190, 220, ch_2_img, 420, 65)
chapter_3_button = button.Button(screen, 190, 320, ch_3_img, 420, 65)
credits_button = button.Button(screen, 660, 350, credits_img, 130, 50)
back_button = button.Button(screen, 710, 300, back_img, 64, 64)

# Function to show credits screen
def show_credits():
    running = True
    while running:
        screen.fill(black)  # Black background for credits
        draw_text("Credits", font, white, 250, 50)
        draw_text("Game by: Alex C., Lauro F., Lara N.", small_font, white, 50, 120)
        draw_text("Music by M'magic Bhomb - @mmagicbhomb", small_font, white, 50, 160)
        draw_text("Additional credits: http://www.codingwithruss.com/", small_font, white, 50, 200)
        
        # Draw back button
        if back_button.draw():
            return  # Exit credits screen
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.update()
        clock.tick(fps)

# Main menu loop
run = True
while run:
    clock.tick(fps)

    # Draw background
    draw_bg()

    # draw title text
    draw_text("The Mürren games", font, blue, 220, 50)

    # Check and draw buttons
    if chapter_2_button.draw():
        subprocess.Popen([sys.executable, 'play_frozen_lake.py'])
    
    if chapter_3_button.draw():
        subprocess.Popen([sys.executable, 'graphic_ch2.py'])

    if chapter_1_button.draw():
        subprocess.Popen([sys.executable, 'graphic_ch2.py'])   # Lauro to change with ur file

    if credits_button.draw():
        show_credits()  # Open credits screen

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    pygame.display.update()

pygame.quit()
