import tkinter as tk

# Maze configuration
MAZE_WIDTH = 10  # Number of tiles horizontally
MAZE_HEIGHT = 10  # Number of tiles vertically
TILE_SIZE = 40  # Size of each tile in pixels

# Maze layout (0 = empty, 1 = wall, 2 = character, 3 = item)
maze = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 3, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

# Character and item positions (derived from the maze)
character_pos = (1, 1)  # (row, column)
item_pos = (8, 8)  # (row, column)

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
    3: "green"   # Item
}

def draw_maze():
    """Draw the maze based on the grid."""
    for row in range(MAZE_HEIGHT):
        for col in range(MAZE_WIDTH):
            x1 = col * TILE_SIZE
            y1 = row * TILE_SIZE
            x2 = x1 + TILE_SIZE
            y2 = y1 + TILE_SIZE
            tile_type = maze[row][col]
            canvas.create_rectangle(x1, y1, x2, y2, fill=COLORS[tile_type], outline="gray")

def update_character_position(new_pos):
    """Update the character's position on the maze."""
    global character_pos
    # Clear the old character position
    maze[character_pos[0]][character_pos[1]] = 0
    # Set the new character position
    character_pos = new_pos
    maze[character_pos[0]][character_pos[1]] = 2
    # Redraw the maze
    draw_maze()

def update_item_position(new_pos):
    """Update the item's position on the maze."""
    global item_pos
    # Clear the old item position
    maze[item_pos[0]][item_pos[1]] = 0
    # Set the new item position
    item_pos = new_pos
    maze[item_pos[0]][item_pos[1]] = 3
    # Redraw the maze
    draw_maze()

# Initial draw
draw_maze()

# Example updates (simulating reinforcement learning steps)
def simulate_reinforcement_learning():
    # Move character to a new position
    update_character_position((2, 2))
    root.after(1000, lambda: update_character_position((3, 3)))
    root.after(2000, lambda: update_character_position((4, 4)))
    # Discover an item
    root.after(3000, lambda: update_item_position((6, 6)))

# Start the simulation
root.after(500, simulate_reinforcement_learning)

# Run the Tkinter event loop
root.mainloop()