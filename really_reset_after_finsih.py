import tkinter as tk
import random
import numpy as np
from collections import deque
from tensorflow.keras import models, layers, optimizers
import os
import json
from datetime import datetime

# Configuration
MAZE_WIDTH = 10
MAZE_HEIGHT = 10
TILE_SIZE = 20

# Colors for different tile types
COLORS = {
    0: "white",  # Empty
    1: "black",  # Wall
    2: "blue",  # Character
    3: "green",  # +reward
    4: "yellow",  # Start
    5: "red",  # Finish
    6: "purple"  # -reward
}


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.min_memory_size = 8
        self.steps = 0
        self.model = self._build_model()

    def _build_model(self):
        """Build the neural network for Q-value approximation."""
        model = models.Sequential()
        model.add(layers.Dense(24, activation='relu', input_shape=(self.state_size,)))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mean_squared_error',
                      optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose an action using epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            print(f"Random action chosen (ε={self.epsilon:.3f}): {action}")
            return action

        q_values = self.model.predict(state, verbose=0)
        action = np.argmax(q_values[0])
        print(f"DQN action chosen (ε={self.epsilon:.3f}): {action}")
        print(f"Q-values: {q_values[0]}")
        return action

    def replay(self, batch_size):
        """Train on a batch of experiences."""
        if len(self.memory) < batch_size:
            print(f"Memory buffer size: {len(self.memory)}/{batch_size}")
            return

        print("\n--- Training Batch ---")
        minibatch = random.sample(self.memory, batch_size)
        total_loss = 0

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_q_values = self.model.predict(next_state, verbose=0)[0]
                target = reward + self.gamma * np.amax(next_q_values)

            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            history = self.model.fit(state, target_f, epochs=1, verbose=0)
            total_loss += history.history['loss'][0]

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        print(f"Loss: {total_loss / batch_size:.4f}, ε: {self.epsilon:.3f}")


class MazeGame:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Maze Game with DQN")

        # Create main canvas
        self.canvas = tk.Canvas(self.root,
                                width=MAZE_WIDTH * TILE_SIZE,
                                height=MAZE_HEIGHT * TILE_SIZE)
        self.canvas.pack(pady=10)

        # Create control panel
        self.create_controls()

        # Initialize game state
        self.maze = [[1 for _ in range(MAZE_WIDTH)] for _ in range(MAZE_HEIGHT)]
        self.start = (1, 1)
        self.finish = (MAZE_WIDTH - 2, MAZE_HEIGHT - 2)
        self.character_pos = self.start
        self.is_moving = False

        # Initialize DQN
        state_size = MAZE_WIDTH * MAZE_HEIGHT
        action_size = 4  # Up, Right, Down, Left
        self.agent = DQNAgent(state_size, action_size)

        # Generate initial maze
        self.generate_maze()
        self.add_items()

        # Store initial state after ALL setup is complete
        self.initial_maze_state = [row[:] for row in self.maze]

        # Place character at start
        self.maze[self.start[1]][self.start[0]] = 2

        self.draw_maze()

    def create_controls(self):
        """Create control buttons."""
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=5)

        tk.Button(control_frame, text="Start DQN",
                  command=self.start_dqn).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Start Random",
                  command=self.start_random).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Reset",
                  command=self.reset_game).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Stop",
                  command=self.stop_movement).pack(side=tk.LEFT, padx=5)

    def generate_maze(self):
        """Generate maze using DFS algorithm."""
        stack = [self.start]
        while stack:
            x, y = stack[-1]
            self.maze[y][x] = 0
            neighbors = []

            for dx, dy in [(0, -2), (2, 0), (0, 2), (-2, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < MAZE_WIDTH and 0 <= ny < MAZE_HEIGHT
                        and self.maze[ny][nx] == 1):
                    neighbors.append((nx, ny))

            if neighbors:
                nx, ny = random.choice(neighbors)
                self.maze[(y + ny) // 2][(x + nx) // 2] = 0
                stack.append((nx, ny))
            else:
                stack.pop()

        # Set start and finish after maze generation
        self.maze[self.start[1]][self.start[0]] = 4  # Start
        self.maze[self.finish[1]][self.finish[0]] = 5  # Finish

    def add_items(self, num_positive=5, num_negative=5):
        """Add reward items to the maze."""
        for _ in range(num_positive):
            while True:
                x, y = random.randint(0, MAZE_WIDTH - 1), random.randint(0, MAZE_HEIGHT - 1)
                if self.maze[y][x] == 0:
                    self.maze[y][x] = 3
                    break

        for _ in range(num_negative):
            while True:
                x, y = random.randint(0, MAZE_WIDTH - 1), random.randint(0, MAZE_HEIGHT - 1)
                if self.maze[y][x] == 0:
                    self.maze[y][x] = 6
                    break

    def draw_maze(self):
        """Draw the current state of the maze."""
        self.canvas.delete("all")
        for row in range(MAZE_HEIGHT):
            for col in range(MAZE_WIDTH):
                x1 = col * TILE_SIZE
                y1 = row * TILE_SIZE
                x2 = x1 + TILE_SIZE
                y2 = y1 + TILE_SIZE
                tile_type = self.maze[row][col]
                self.canvas.create_rectangle(x1, y1, x2, y2,
                                             fill=COLORS[tile_type],
                                             outline="gray")
        self.root.update()

    def get_state(self):
        """Convert current maze state to DQN input."""
        state = np.zeros((MAZE_WIDTH, MAZE_HEIGHT))
        state[self.character_pos[1]][self.character_pos[0]] = 1
        return state.flatten().reshape(1, -1)

    def is_valid_move(self, x, y):
        """Check if a move is valid."""
        return (0 <= x < MAZE_WIDTH and
                0 <= y < MAZE_HEIGHT and
                self.maze[y][x] != 1)

    def move_character_dqn(self):
        """Move character using DQN agent."""
        if not self.is_moving:
            return

        try:
            print(f"\n=== DQN Movement Step {self.agent.steps} ===")
            print(f"Current position: {self.character_pos}")

            state = self.get_state()
            action = self.agent.act(state)
            directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
            dx, dy = directions[action]
            new_x, new_y = self.character_pos[0] + dx, self.character_pos[1] + dy

            print(f"Attempting move to: ({new_x}, {new_y})")

            if self.is_valid_move(new_x, new_y):
                print("Move is valid")
                # Store what's at the new position
                new_pos_type = self.maze[new_y][new_x]

                # Clear old position (character's current position)
                self.maze[self.character_pos[1]][self.character_pos[0]] = 0

                # Calculate reward based on what we're moving to
                reward = -0.1
                done = False

                if new_pos_type == 3:  # Positive reward
                    reward = 1.0
                    print("Found positive reward! +1.0")
                elif new_pos_type == 6:  # Negative reward
                    reward = -1.0
                    print("Hit negative reward! -1.0")
                elif (new_x, new_y) == self.finish:  # Reached finish
                    reward = 10.0
                    print("Reached the finish! +10.0")
                    done = True

                # Update position and place character
                self.character_pos = (new_x, new_y)
                self.maze[new_y][new_x] = 2
                print(f"Move completed. Reward: {reward}")

                next_state = self.get_state()
                self.agent.remember(state, action, reward, next_state, done)
                self.agent.steps += 1
                self.agent.replay(self.agent.min_memory_size)

                self.draw_maze()

                if done:
                    self.handle_success()

            else:
                print("Invalid move - hit wall or boundary")
                reward = -0.5
                next_state = state
                self.agent.remember(state, action, reward, next_state, False)
                self.agent.steps += 1
                self.agent.replay(self.agent.min_memory_size)

            if self.is_moving:
                self.root.after(100, self.move_character_dqn)

        except Exception as e:
            print(f"Error in move_character_dqn: {str(e)}")
            self.is_moving = False

    def move_character_random(self):
        """Move character randomly."""
        if not self.is_moving:
            return

        x, y = self.character_pos
        valid_moves = []
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            if self.is_valid_move(new_x, new_y):
                valid_moves.append((new_x, new_y))

        if valid_moves:
            self.maze[y][x] = 0
            new_x, new_y = random.choice(valid_moves)
            self.character_pos = (new_x, new_y)

            if self.maze[new_y][new_x] == 3:
                print("Found positive reward!")
            elif self.maze[new_y][new_x] == 6:
                print("Hit negative reward!")
            elif (new_x, new_y) == self.finish:
                print("Reached the finish!")
                self.is_moving = False

            self.maze[new_y][new_x] = 2
            self.draw_maze()

            if self.is_moving:
                self.root.after(100, self.move_character_random)

    def handle_success(self):
        """Handle successful completion of the maze."""
        # Save the current successful state
        self.save_success()

        # Store the original tile types
        stored_maze = [row[:] for row in self.maze]  # Deep copy the maze

        # Reset character to start position
        self.maze[self.character_pos[1]][self.character_pos[0]] = stored_maze[self.character_pos[1]][
            self.character_pos[0]]  # Restore the original tile
        self.character_pos = self.start
        self.maze[self.start[1]][self.start[0]] = 2  # Place character at start

        # Make sure start and finish are correctly marked
        self.maze[self.start[1]][self.start[0]] = 2  # Character
        self.maze[self.finish[1]][self.finish[0]] = 5  # Finish

        # Slightly reduce exploration to capitalize on learned path
        self.agent.epsilon = max(self.agent.epsilon * 0.95, self.agent.epsilon_min)
        print(f"\nReset position for additional training. Adjusted epsilon to: {self.agent.epsilon:.3f}")

        # Update display
        self.draw_maze()

        # Continue movement
        self.is_moving = True

    def start_dqn(self):
        """Start DQN-based movement."""
        if not self.is_moving:
            self.is_moving = True
            self.move_character_dqn()

    def start_random(self):
        """Start random movement."""
        if not self.is_moving:
            self.is_moving = True
            self.move_character_random()

    def stop_movement(self):
        """Stop all movement."""
        self.is_moving = False

    def reset_game(self):
        """Reset the game state."""
        self.is_moving = False
        self.maze = [[1 for _ in range(MAZE_WIDTH)] for _ in range(MAZE_HEIGHT)]
        self.character_pos = self.start
        self.generate_maze()
        self.add_items()
        self.draw_maze()

    def save_success(self):
        """Save the successful maze and trained model."""
        os.makedirs('successful_runs', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save maze configuration
        maze_data = {
            'maze': self.maze,
            'start': self.start,
            'finish': self.finish,
            'steps': self.agent.steps
        }
        maze_file = f'successful_runs/maze_{timestamp}.json'
        with open(maze_file, 'w') as f:
            json.dump(maze_data, f)
        print(f"Maze saved to {maze_file}")

        # Save model
        model_file = f'successful_runs/model_{timestamp}.h5'
        self.agent.model.save(model_file)
        print(f"Model saved to {model_file}")

        # Save stats
        stats = {
            'total_steps': self.agent.steps,
            'final_epsilon': self.agent.epsilon,
            'memory_size': len(self.agent.memory)
        }
        stats_file = f'successful_runs/stats_{timestamp}.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f)
        print(f"Stats saved to {stats_file}")

    def run(self):
        """Start the game main loop."""
        self.root.mainloop()


# Main entry point
if __name__ == "__main__":
    game = MazeGame()
    game.run()
