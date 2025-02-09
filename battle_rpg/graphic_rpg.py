import pygame

pygame.init()

clock = pygame.time.Clock()
fps= 60

# pygame window
bottom_panel = 150
screen_width = 800
screen_heigth = 400 + bottom_panel

screen = pygame.display.set_mode((screen_width, screen_heigth))
pygame.display.set_caption('Murren final boss battle')

# load images
# Game background
background_image=pygame.image.load('img/Background/background.jpg').convert_alpha()
# panel image
panel_image=pygame.image.load('img/Icons/panel_double.png').convert_alpha()


# function for drawing image
def draw_bg():
    screen.blit(background_image, (0,0))

# function to draw panel
def draw_panel():
    screen.blit(panel_image, (0,screen_heigth- bottom_panel))


# Fighter class
class Fighter():
    def __init__(self, x, y, name, max_hp, strnght, potions):
        self.name=name
        self.max_hp=max_hp
        self.hp=max_hp
        self.strenght=strnght
        self.start_potions=potions
        self.potions=potions
        self.alive = True
        img=pygame.image.load(f'img/{self.name}/Idle/0.png')
        self.image=pygame.transform.scale(img, (img.get_width()*3, img.get_height()*3))
        self.rect=self.image.get_rect()
        self.rect.center=(x,y)

    def draw(self):
        screen.blit(self.image, self.rect)


knight=Fighter(200, 260, 'Knight', 30, 10, 3)
bandit1 = Fighter(550, 270, 'Bandit', 20, 6, 1)
bandit2 = Fighter(700, 270, 'Bandit', 20, 6, 1)

bandit_list= []
bandit_list.append(bandit1)
bandit_list.append(bandit2)


run = True
while run:

    clock.tick(fps)

    # draw background
    draw_bg()
    # draw panel
    draw_panel()

    # draw fighters
    knight.draw()
    for bandit in bandit_list:
        bandit.draw()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    pygame.display.update()

pygame.quit()