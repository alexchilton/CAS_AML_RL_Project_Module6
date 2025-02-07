import tkinter as tk
import random
import numpy as np

# Maze configuration
MAZE_WIDTH = 20  # Number of tiles horizontally
MAZE_HEIGHT = 20  # Number of tiles vertically
TILE_SIZE = 20  # Size of each tile in pixels

# Maze layout (0 = empty, 1 = wall, 2 = character, 3 = item, 4 = start, 5 = finish)
maze = [[1 for _ in range(MAZE_WIDTH)] for _ in range(MAZE_HEIGHT)]

# Directions for movement
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, Right, Down, Left

# Initialize Tkinter
root = tk.Tk()
root.title("RPG Maze Visualization")
canvas = tk.Canvas(root, width=MAZE_WIDTH * TILE_SIZE, height=MAZE_HEIGHT * TILE_SIZE)
canvas.pack()

# Colors for tiles
COLORS = {
    0: "white",  # Empty
    1: "black",  # Wall
    2: "blue",   # Character
    3: "green",  # Item
    4: "yellow", # Start
    5: "red"     # Finish
}

# Generate a maze using Depth-First Search
def generate_maze(maze, start, finish):
    stack = [start]
    while stack:
        x, y = stack[-1]
        maze[y][x] = 0  # Mark as empty
        neighbors = []
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx * 2, y + dy * 2
            if 0 <= nx < MAZE_WIDTH and 0 <= ny < MAZE_HEIGHT and maze[ny][nx] == 1:
                neighbors.append((nx, ny))
        if neighbors:
            nx, ny = random.choice(neighbors)
            maze[(y + ny) // 2][(x + nx) // 2] = 0  # Break the wall
            stack.append((nx, ny))
        else:
            stack.pop()
    maze[start[1]][start[0]] = 4  # Start
    maze[finish[1]][finish[0]] = 5  # Finish

# Generate the maze
start = (1, 1)
finish = (MAZE_WIDTH - 2, MAZE_HEIGHT - 2)
generate_maze(maze, start, finish)

# Add items to the maze
def add_items(maze, num_items=5):
    for _ in range(num_items):
        while True:
            x, y = random.randint(0, MAZE_WIDTH - 1), random.randint(0, MAZE_HEIGHT - 1)
            if maze[y][x] == 0:  # Place item only on empty tiles
                maze[y][x] = 3
                break

add_items(maze, num_items=10)  # Add 10 items to the maze

# Character position
character_pos = start

def draw_maze():
    """Draw the maze based on the grid."""
    canvas.delete("all")
    for row in range(MAZE_HEIGHT):
        for col in range(MAZE_WIDTH):
            x1 = col * TILE_SIZE
            y1 = row * TILE_SIZE
            x2 = x1 + TILE_SIZE
            y2 = y1 + TILE_SIZE
            tile_type = maze[row][col]
            canvas.create_rectangle(x1, y1, x2, y2, fill=COLORS[tile_type], outline="gray")

def is_valid_move(x, y):
    """Check if the move is valid (within bounds and not a wall)."""
    return 0 <= x < MAZE_WIDTH and 0 <= y < MAZE_HEIGHT and maze[y][x] != 1

def move_character_randomly():
    """Move the character randomly to a valid adjacent tile."""
    global character_pos
    x, y = character_pos
    # Get valid moves
    valid_moves = [(x + dx, y + dy) for dx, dy in DIRECTIONS if is_valid_move(x + dx, y + dy)]
    if valid_moves:
        new_x, new_y = random.choice(valid_moves)
        # Update character position
        maze[y][x] = 0  # Clear old position
        character_pos = (new_x, new_y)
        if maze[new_y][new_x] == 3:  # Item found
            print("Item found at:", (new_x, new_y))
        maze[new_y][new_x] = 2  # Set new position
        draw_maze()
        # Check if the character reached the finish
        if (new_x, new_y) == finish:
            print("Character reached the finish!")
            return
    # Schedule the next move
    root.after(200, move_character_randomly)

# Initial draw
draw_maze()

# Start the random movement
root.after(500, move_character_randomly)

# Print the maze as a numpy array for visualization
print("Maze as a numpy array:")
print(np.array(maze))

# Run the Tkinter event loop
root.mainloop()