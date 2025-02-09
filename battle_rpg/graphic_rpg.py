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

# define font
font= pygame.font.SysFont('Times New Roman', 21)

# define color
red=(255, 0, 0)
green= (0, 255, 0)
blue= (0, 0, 255)
powderblue= (176,224,230)


# load images
# Game background
background_image=pygame.image.load('img/Background/background.jpg').convert_alpha()
# panel image
panel_image=pygame.image.load('img/Icons/panel_double.png').convert_alpha()


# create function for drawing text
def draw_text(text, font, text_col, x, y):
    img=font.render(text, True, text_col)
    screen.blit(img, (x, y))


# function for drawing image
def draw_bg():
    screen.blit(background_image, (0,0))

# function to draw panel
def draw_panel():
    # drw panel rectangle
    screen.blit(panel_image, (0,screen_heigth- bottom_panel))
    # show knight stats
    draw_text(f'{knight.name} HP: {knight.hp}', font, blue, 120, screen_heigth-bottom_panel +10 )
    for count, i in enumerate(bandit_list):
        # show name and health
        draw_text(f'{i.name} HP: {i.hp}', font, powderblue, 530, (screen_heigth-bottom_panel +10 )+count*50)


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
        self.animation_list=[]
        self.frame_index =0
        self.action = 0 # 0: Idle , 1: attack, 2: Hurt, 3: Death
        self.update_time=pygame.time.get_ticks()
        # load idle images
        temp_list=[]
        for i in range(8):
            img=pygame.image.load(f'img/{self.name}/Idle/{i}.png')
            img=pygame.transform.scale(img, (img.get_width()*3, img.get_height()*3))
            temp_list.append(img)
        self.animation_list.append(temp_list)
        # load attack images
        temp_list=[]
        for i in range(8):
            img=pygame.image.load(f'img/{self.name}/Attack/{i}.png')
            img=pygame.transform.scale(img, (img.get_width()*3, img.get_height()*3))
            temp_list.append(img)
        self.animation_list.append(temp_list)
        self.image=self.animation_list[self.action][self.frame_index]
        self.rect=self.image.get_rect()
        self.rect.center=(x,y)

    def update(self):
        animation_cooldown = 100
        # handle animation
        # update image
        self.image=self.animation_list[self.action][self.frame_index]
        # check if enough time hasd passed after pastupdate
        if pygame.time.get_ticks() - self.update_time > animation_cooldown:
            self.update_time = pygame.time.get_ticks()
            self.frame_index += 1
        # if the animation has run out then reset back to the start
        if self.frame_index >= len(self.animation_list[self.action]):
            self.frame_index=0

    def draw(self):
        screen.blit(self.image, self.rect)


class HealthBar():
    def __init__(self, x, y, hp, max_hp):
        self.x = x 
        self.y = y
        self.hp = hp
        self.max_hp = max_hp

    def draw(self, hp):
        # update with new health
        self.hp = hp
        # calculate health ration
        ratio = self.hp / self.max_hp
        pygame.draw.rect(screen, red, (self.x, self.y, 150, 15))
        pygame.draw.rect(screen, green, (self.x, self.y, 150*ratio, 15))


knight=Fighter(200, 260, 'Knight', 30, 10, 3)
bandit1 = Fighter(550, 270, 'Bandit', 20, 6, 1)
bandit2 = Fighter(700, 270, 'Bandit', 20, 6, 1)

bandit_list= []
bandit_list.append(bandit1)
bandit_list.append(bandit2)

knight_health_bar=HealthBar(120, screen_heigth-bottom_panel+40, knight.hp, knight.max_hp)
bandit1_health_bar=HealthBar(530, screen_heigth-bottom_panel+40, bandit1.hp, bandit1.max_hp)
bandit2_health_bar=HealthBar(530, screen_heigth-bottom_panel+90, bandit2.hp, bandit2.max_hp)


run = True
while run:

    clock.tick(fps)

    # draw background
    draw_bg()
    # draw panel
    draw_panel()
    knight_health_bar.draw(knight.hp)
    bandit1_health_bar.draw(bandit1.hp)
    bandit2_health_bar.draw(bandit2.hp)

    # draw fighters
    knight.update()
    knight.draw()
    for bandit in bandit_list:
        bandit.update()
        bandit.draw()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    pygame.display.update()

pygame.quit()