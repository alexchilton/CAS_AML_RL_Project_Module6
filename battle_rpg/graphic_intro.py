import pygame
import subprocess
import sys
import os

# Initialize Pygame
pygame.init()

# Screen setup
WIDTH, HEIGHT = 800, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("The Mürren Games")

# Load images and resize them
images = [
    pygame.image.load("img/Background/intro_3_s.jpg"),
    pygame.image.load("img/Background/intro_3_s.jpg"),
    pygame.image.load("img/Background/intro_1_s.jpg"),
    pygame.image.load("img/Background/intro_2_s.jpg"),
    pygame.image.load("img/Background/intro_4_s.jpg"), 
    pygame.image.load("img/Background/intro_5_s.jpg")  
]
images = [pygame.transform.scale(img, (WIDTH, HEIGHT)) for img in images]

# Apply slight transparency to background images (75% opacity)
for i in range(len(images)):
    temp = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    temp.fill((255, 255, 255, 64))  # White with 25% opacity (slight dimming)
    images[i] = images[i].convert_alpha()
    images[i].blit(temp, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

# Load and play background music (looping)
pygame.mixer.music.load("music/M_magic_Bhomb_ThePathofAHero.mp3")
pygame.mixer.music.set_volume(0.5)  # Set volume (0.0 to 1.0)
pygame.mixer.music.play(-1)  # -1 makes it loop indefinitely

# Font setup
font = pygame.font.Font(None, 28)  # Regular font size for text
title_font = pygame.font.Font(None, 72)  # Larger font for the title
button_font = pygame.font.Font(None, 36)  # Font for the main menu button

# Text content - one set of lines per image
image_text = [
    [
        "THE MÜRREN GAMES"
    ],
    [
        "January 2025, Mürren...",
        "A new and unexpected adventure awaits a young CAS student,",
        "who will face challenges not alone, but with the help of valuable Agents!"
    ],
    [
        "Frozen lakes must be crossed..."
    ],
    [
        "Fearsome bandits, aggressively guarding Mykhailo, await your arrival..."
    ],
    [
        "But first, the best route to Mürren must be found..."
    ],
    [
        "All of this, powered by RL (and some beers! :))",
        "",
        "Good luck, young adventurer.", 
        "",
        "May the code be with you!"
    ]
]

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GOLD = (255, 215, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
POWDERBLUE = (176, 224, 230)

def render_text_line_with_outline(surface, text, font, pos, text_color=WHITE, outline_color=BLACK, outline_width=2):
    """Render a single line of text with an outline for better visibility"""
    # Render the outline by offsetting the text in multiple directions
    for dx, dy in [(ox, oy) for ox in range(-outline_width, outline_width + 1) 
                          for oy in range(-outline_width, outline_width + 1)
                          if ox != 0 or oy != 0]:
        outline_surf = font.render(text, True, outline_color)
        outline_rect = outline_surf.get_rect(center=(pos[0] + dx, pos[1] + dy))
        surface.blit(outline_surf, outline_rect)
    
    # Render the main text
    text_surf = font.render(text, True, text_color)
    text_rect = text_surf.get_rect(center=pos)
    surface.blit(text_surf, text_rect)
    
    return text_rect  # Return the rect of the rendered text

def render_growing_title(surface, text, font, pos, scale_factor, text_color=POWDERBLUE, outline_color=BLACK, outline_width=3):
    """Render title text with growth effect"""
    # Apply scaling to the font size
    scaled_size = int(font.get_height() * scale_factor)
    if scaled_size < 20:  # Minimum size
        scaled_size = 20
    
    # Use default font instead of trying to get the name
    temp_font = pygame.font.Font(None, scaled_size)
    
    # Render with outline
    text_rect = render_text_line_with_outline(
        surface, 
        text, 
        temp_font, 
        pos, 
        text_color, 
        outline_color, 
        outline_width
    )
    
    return text_rect

def render_multiline_text(surface, text_lines, font, start_pos, line_spacing=10, **kwargs):
    """Render multiple lines of text with outline"""
    current_y = start_pos[1] - ((len(text_lines) - 1) * (font.get_height() + line_spacing)) // 2
    rects = []
    
    for line in text_lines:
        text_rect = render_text_line_with_outline(
            surface, 
            line, 
            font, 
            (start_pos[0], current_y), 
            **kwargs
        )
        rects.append(text_rect)
        current_y += text_rect.height + line_spacing
    
    return rects

def create_button(text, font, x, y, padding=10, text_color=WHITE, bg_color=(100, 100, 100), hover_color=(150, 150, 150)):
    """Create a button with text"""
    text_surf = font.render(text, True, text_color)
    text_rect = text_surf.get_rect()
    
    # Create button rect with padding
    button_rect = pygame.Rect(0, 0, text_rect.width + padding*2, text_rect.height + padding*2)
    button_rect.center = (x, y)
    
    # Position text in the center of the button
    text_rect.center = button_rect.center
    
    return {
        'text': text,
        'text_surf': text_surf,
        'text_rect': text_rect,
        'rect': button_rect,
        'color': bg_color,
        'hover_color': hover_color,
        'is_hovered': False
    }

def draw_button(surface, button):
    """Draw a button on the surface"""
    color = button['hover_color'] if button['is_hovered'] else button['color']
    
    # Draw button background with rounded corners
    pygame.draw.rect(surface, color, button['rect'], border_radius=10)
    
    # Draw button outline
    pygame.draw.rect(surface, WHITE, button['rect'], width=2, border_radius=10)
    
    # Draw button text
    surface.blit(button['text_surf'], button['text_rect'])

def animate_title_sequence(image):
    """Animated sequence for the title slide"""
    fade_surface = pygame.Surface((WIDTH, HEIGHT))
    fade_surface.fill((0, 0, 0))
    
    # Fade in background
    for alpha in range(0, 256, 5):  
        fade_surface.set_alpha(255 - alpha)
        screen.blit(image, (0, 0))
        screen.blit(fade_surface, (0, 0))
        pygame.display.update()
        pygame.time.delay(30)
        
        # Check for exit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return False
    
    # Title animation
    title_text = "THE MÜRREN GAMES"
    start_scale = 0.1
    end_scale = 1.5
    steps = 60
    
    for i in range(steps + 30):  # Animation + hold
        scale = start_scale
        if i < steps:
            # Ease-in-out scaling
            progress = i / steps
            scale = start_scale + (end_scale - start_scale) * (progress * (2 - progress))
        else:
            scale = end_scale
            
        screen.blit(image, (0, 0))
        
        # Draw the growing title
        render_growing_title(
            screen, 
            title_text, 
            title_font, 
            (WIDTH // 2, HEIGHT // 2), 
            scale
        )
        
        pygame.display.update()
        pygame.time.delay(30)
        
        # Check for exit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return False
            elif event.type == pygame.KEYDOWN and event.key != pygame.K_ESCAPE:
                return True
    
    # Fade out
    for alpha in range(0, 256, 5):  
        fade_surface.set_alpha(alpha)
        screen.blit(image, (0, 0))
        
        # Draw title at full size
        render_growing_title(
            screen, 
            title_text, 
            title_font, 
            (WIDTH // 2, HEIGHT // 2), 
            end_scale
        )
        
        screen.blit(fade_surface, (0, 0))
        pygame.display.update()
        pygame.time.delay(30)
        
        # Check for exit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return False
            elif event.type == pygame.KEYDOWN and event.key != pygame.K_ESCAPE:
                return True
    
    return True

def fade_in_out(image, text_lines, fade_speed=5, is_final=False, is_second_slide=False):
    """Handles fade-in and fade-out effect with multiline text overlay"""
    fade_surface = pygame.Surface((WIDTH, HEIGHT))
    fade_surface.fill((0, 0, 0))
    
    # Calculate center position for text
    text_pos = (WIDTH // 2, HEIGHT // 2)
    
    # Create main menu button if this is the final slide
    main_menu_button = None
    if is_final:
        main_menu_button = create_button(
            "Main Menu", 
            button_font, 
            WIDTH - 100, 
            HEIGHT - 50, 
            padding=15, 
            bg_color=(80, 0, 0), 
            hover_color=(120, 0, 0)
        )
    
    # Fade in
    for alpha in range(0, 256, fade_speed):  
        fade_surface.set_alpha(255 - alpha)  # Reduce black overlay
        screen.blit(image, (0, 0))
        
        # Draw multiline text with outline
        render_multiline_text(screen, text_lines, font, text_pos)
        
        # Draw main menu button if on final slide
        if is_final and main_menu_button:
            draw_button(screen, main_menu_button)
        
        screen.blit(fade_surface, (0, 0))
        pygame.display.update()
        pygame.time.delay(30)
        
        # Check for exit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return False
    
    # Hold image with text - longer for second slide
    hold_duration = 6000 if is_second_slide else 3000  # 6 seconds for second slide, 3 for others
    hold_start = pygame.time.get_ticks()
    
    while pygame.time.get_ticks() - hold_start < hold_duration:
        screen.blit(image, (0, 0))
        
        # Draw multiline text with outline
        render_multiline_text(screen, text_lines, font, text_pos)
        
        # Draw and handle main menu button if on final slide
        if is_final and main_menu_button:
            # Check mouse position for hover effect
            mouse_pos = pygame.mouse.get_pos()
            main_menu_button['is_hovered'] = main_menu_button['rect'].collidepoint(mouse_pos)
            
            # Draw the button
            draw_button(screen, main_menu_button)
        
        pygame.display.update()
        
        # Check for events
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return False
            # Skip to next slide on any key press except ESC
            elif event.type == pygame.KEYDOWN and event.key != pygame.K_ESCAPE:
                return True
            # Check for button click if on final slide
            elif is_final and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if main_menu_button and main_menu_button['rect'].collidepoint(event.pos):
                    # Launch the main menu
                    try:
                        pygame.mixer.music.fadeout(500)
                        pygame.quit()
                        if sys.platform.startswith('win'):
                            os.system('python graphic_main_menu.py')
                        else:
                            subprocess.run(['python', 'graphic_main_menu.py'])
                        return False  # Exit intro
                    except Exception as e:
                        print(f"Error launching main menu: {e}")
                        return False
    
    # Fade out
    for alpha in range(0, 256, fade_speed):  
        fade_surface.set_alpha(alpha)  # Increase black overlay
        screen.blit(image, (0, 0))
        
        # Draw multiline text with outline
        render_multiline_text(screen, text_lines, font, text_pos)
        
        # Draw main menu button if on final slide
        if is_final and main_menu_button:
            # Check mouse position for hover effect
            mouse_pos = pygame.mouse.get_pos()
            main_menu_button['is_hovered'] = main_menu_button['rect'].collidepoint(mouse_pos)
            
            # Draw the button
            draw_button(screen, main_menu_button)
        
        screen.blit(fade_surface, (0, 0))
        pygame.display.update()
        pygame.time.delay(30)
        
        # Check for exit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return False
            # Skip to next slide on any key press except ESC
            elif event.type == pygame.KEYDOWN and event.key != pygame.K_ESCAPE:
                return True
            # Check for button click if on final slide
            elif is_final and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if main_menu_button and main_menu_button['rect'].collidepoint(event.pos):
                    # Launch the main menu
                    try:
                        pygame.mixer.music.fadeout(500)
                        pygame.quit()
                        if sys.platform.startswith('win'):
                            os.system('python graphic_main_menu.py')
                        else:
                            subprocess.run(['python', 'graphic_main_menu.py'])
                        return False  # Exit intro
                    except Exception as e:
                        print(f"Error launching main menu: {e}")
                        return False
    
    return True

def run_intro():
    running = True
    
    # Start with title animation
    if not animate_title_sequence(images[0]):
        running = False
    
    # Skip the first image since we already showed it in the title animation
    current_image = 1
    
    while running and current_image < len(images):
        # Process events before the fade
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False  # Exit intro on ESC press
        
        if not running:
            break
            
        # Check if this is the final slide or second slide
        is_final = (current_image == len(images) - 1)
        is_second_slide = (current_image == 1)  # Index 1 is the second slide with long text
            
        # Fade in/out the current image with its corresponding multiline text
        if not fade_in_out(images[current_image], image_text[current_image], is_final=is_final, is_second_slide=is_second_slide):
            running = False
            break
            
        current_image += 1  # Move to next image
    
    # If we reached the end, show main menu button and wait for click
    if running and current_image >= len(images):
        # Create the main menu button
        main_menu_button = create_button(
            "Main Menu", 
            button_font, 
            WIDTH - 100, 
            HEIGHT - 50, 
            padding=15, 
            bg_color=(80, 0, 0), 
            hover_color=(120, 0, 0)
        )
        
        waiting = True
        while waiting:
            screen.blit(images[-1], (0, 0))
            
            # Draw the last text
            render_multiline_text(screen, image_text[-1], font, (WIDTH // 2, HEIGHT // 2))
            
            # Check mouse position for hover effect
            mouse_pos = pygame.mouse.get_pos()
            main_menu_button['is_hovered'] = main_menu_button['rect'].collidepoint(mouse_pos)
            
            # Draw the button
            draw_button(screen, main_menu_button)
            
            pygame.display.update()
            
            # Check for events
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    waiting = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if main_menu_button['rect'].collidepoint(event.pos):
                        # Launch the main menu
                        try:
                            pygame.mixer.music.fadeout(500)
                            pygame.quit()
                            if sys.platform.startswith('win'):
                                os.system('python graphic_main_menu.py')
                            else:
                                subprocess.run(['python', 'graphic_main_menu.py'])
                            return  # Exit function
                        except Exception as e:
                            print(f"Error launching main menu: {e}")
                            waiting = False
    
    pygame.mixer.music.fadeout(1000)  # Fade out music when intro ends

# Run the intro
run_intro()

# Quit Pygame
pygame.quit()