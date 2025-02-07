import tkinter as tk
import random
import numpy as np
from collections import deque
from tensorflow.keras import models, layers, optimizers

# Maze configuration
MAZE_WIDTH = 20  # Number of tiles horizontally
MAZE_HEIGHT = 20  # Number of tiles vertically
TILE_SIZE = 20  # Size of each tile in pixels

# Maze layout (0 = empty, 1 = wall, 2 = character, 3 = +reward item, 4 = start, 5 = finish, 6 = -reward item)
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
    3: "green",  # +reward item
    4: "yellow", # Start
    5: "red",    # Finish
    6: "purple"  # -reward item
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

# Add items to the maze
def add_items(maze, num_positive=5, num_negative=5):
    """Add positive and negative reward items to the maze."""
    for _ in range(num_positive):
        while True:
            x, y = random.randint(0, MAZE_WIDTH - 1), random.randint(0, MAZE_HEIGHT - 1)
            if maze[y][x] == 0:  # Place item only on empty tiles
                maze[y][x] = 3  # Positive reward item
                break
    for _ in range(num_negative):
        while True:
            x, y = random.randint(0, MAZE_WIDTH - 1), random.randint(0, MAZE_HEIGHT - 1)
            if maze[y][x] == 0:  # Place item only on empty tiles
                maze[y][x] = 6  # Negative reward item
                break

# Generate the maze
start = (1, 1)
finish = (MAZE_WIDTH - 2, MAZE_HEIGHT - 2)
generate_maze(maze, start, finish)
add_items(maze, num_positive=5, num_negative=5)  # Add items

# Character position
character_pos = start

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Experience replay buffer
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        """Build the neural network for Q-value approximation."""
        model = models.Sequential()
        model.add(layers.Input(shape=(self.state_size,)))  # Use Input layer
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))  # Use learning_rate
        return model

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose an action using epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        """Train the model using experience replay."""
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Initialize the DQN agent
state_size = MAZE_WIDTH * MAZE_HEIGHT  # Flatten the maze as the state
action_size = 4  # Four possible actions (up, right, down, left)
agent = DQNAgent(state_size, action_size)

# Function to convert maze state to a flattened array
def get_state(maze, character_pos):
    state = np.zeros((MAZE_WIDTH, MAZE_HEIGHT))
    state[character_pos[1]][character_pos[0]] = 1  # Mark character position
    return state.flatten().reshape(1, -1)

# Function to move the character randomly
def move_character_randomly():
    global character_pos
    x, y = character_pos
    # Get valid moves
    valid_moves = [(x + dx, y + dy) for dx, dy in DIRECTIONS if is_valid_move(x + dx, y + dy)]
    if valid_moves:
        new_x, new_y = random.choice(valid_moves)
        # Update character position
        maze[y][x] = 0  # Clear old position
        character_pos = (new_x, new_y)
        if maze[new_y][new_x] == 3:  # Positive reward item
            print("Collected +reward item!")
        elif maze[new_y][new_x] == 6:  # Negative reward item
            print("Collected -reward item!")
        maze[new_y][new_x] = 2  # Set new position
        # Check if the character reached the finish
        if (new_x, new_y) == finish:
            print("Character reached the finish!")
            return
    # Schedule the next move
    root.after(200, move_character_randomly)

# Function to move the character using the DQN agent
def move_character_dqn():
    global character_pos
    state = get_state(maze, character_pos)
    action = agent.act(state)
    dx, dy = DIRECTIONS[action]
    new_x, new_y = character_pos[0] + dx, character_pos[1] + dy
    if is_valid_move(new_x, new_y):
        # Update character position
        maze[character_pos[1]][character_pos[0]] = 0  # Clear old position
        character_pos = (new_x, new_y)
        # Check for items
        if maze[new_y][new_x] == 3:  # Positive reward item
            reward = 10
            print("Collected +reward item!")
        elif maze[new_y][new_x] == 6:  # Negative reward item
            reward = -10
            print("Collected -reward item!")
        else:
            reward = -1  # Small penalty for each move
        maze[new_y][new_x] = 2  # Set new position
        # Check if the character reached the finish
        done = (new_x, new_y) == finish
        if done:
            reward = 100  # Large reward for reaching the finish
            print("Character reached the finish!")
        # Get the next state
        next_state = get_state(maze, character_pos)
        # Remember the experience
        agent.remember(state, action, reward, next_state, done)
        # Train the agent
        agent.replay(32)
        # Redraw the maze
        draw_maze()
        if not done:
            root.after(200, move_character_dqn)

# Function to check if a move is valid
def is_valid_move(x, y):
    """Check if the move is valid (within bounds and not a wall)."""
    return 0 <= x < MAZE_WIDTH and 0 <= y < MAZE_HEIGHT and maze[y][x] != 1

# Function to draw the maze
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

# Toggle between random and DQN movement
use_dqn = False  # Set to False to use random movement

def move_character():
    if use_dqn:
        move_character_dqn()
    else:
        move_character_randomly()

# Initial draw
draw_maze()

# Start the movement
root.after(500, move_character)

# Print the maze as a numpy array for visualization
print("Maze as a numpy array:")
print(np.array(maze))

# Run the Tkinter event loop
root.mainloop()