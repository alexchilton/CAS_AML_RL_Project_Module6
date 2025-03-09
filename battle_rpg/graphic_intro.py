import pygame

# Initialize Pygame
pygame.init()

# Screen setup
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Game Intro")

# Load images and resize them
images = [
    pygame.image.load("image1.jpg"),
    pygame.image.load("image2.jpg"),
    pygame.image.load("image3.jpg")
]
images = [pygame.transform.scale(img, (WIDTH, HEIGHT)) for img in images]

# Load and play background music (looping)
pygame.mixer.music.load("background_music.mp3")
pygame.mixer.music.set_volume(0.5)  # Set volume (0.0 to 1.0)
pygame.mixer.music.play(-1)  # -1 makes it loop indefinitely

# Font setup
font = pygame.font.Font(None, 36)
text_lines = [
    "Long ago, in a forgotten land...",
    "A hero was born, destined for greatness...",
    "But darkness looms over the kingdom..."
]

# Define colors
WHITE = (255, 255, 255)

def fade_in_out(image, fade_speed=5):
    """Handles fade-in and fade-out effect"""
    fade_surface = pygame.Surface((WIDTH, HEIGHT))
    fade_surface.fill((0, 0, 0))
    
    # Fade in
    for alpha in range(0, 256, fade_speed):  
        fade_surface.set_alpha(255 - alpha)  # Reduce black overlay
        screen.blit(image, (0, 0))
        screen.blit(fade_surface, (0, 0))
        pygame.display.update()
        pygame.time.delay(30)
    
    pygame.time.delay(1000)  # Hold image before fading out

    # Fade out
    for alpha in range(0, 256, fade_speed):  
        fade_surface.set_alpha(alpha)  # Increase black overlay
        screen.blit(image, (0, 0))
        screen.blit(fade_surface, (0, 0))
        pygame.display.update()
        pygame.time.delay(30)

def run_intro():
    running = True
    clock = pygame.time.Clock()
    text_y = HEIGHT  # Start text below the screen
    text_speed = 1  # Speed of text movement
    current_image = 0
    
    while running:
        # Fade in/out the current image
        fade_in_out(images[current_image])
        current_image = (current_image + 1) % len(images)  # Cycle images

        # Moving text effect
        screen.blit(images[current_image], (0, 0))  # Draw background
        y_offset = text_y
        for line in text_lines:
            text_surface = font.render(line, True, WHITE)
            screen.blit(text_surface, (50, y_offset))
            y_offset += 40
        
        text_y -= text_speed  # Move text up
        
        pygame.display.update()
        clock.tick(30)  # Control frame rate

        # Event handling (Allow skipping)
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                running = False  # Exit intro on key press

    pygame.mixer.music.stop()  # Stop music when intro ends

# Run the intro
run_intro()

# Quit Pygame
pygame.quit()
