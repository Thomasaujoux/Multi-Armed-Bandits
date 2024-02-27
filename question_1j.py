
import numpy as np
import random


### IMPLEMENTATION OF THE GAME

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)  # 0 for empty, 1 for X, -1 for O
        self.current_player = 1  # Player 1 starts (X)
        self.game_over = False
        self.winner = None

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        return self.board.flatten()

    def step(self, action):
        row, col = divmod(action, 3)
        if self.board[row, col] == 0 and not self.game_over:
            self.board[row, col] = self.current_player
            if self.check_win():
                self.game_over = True
                self.winner = self.current_player
                reward = 1
            elif np.all(self.board != 0):
                self.game_over = True
                self.winner = 0  # Draw
                reward = 0.5
            else:
                self.current_player *= -1
                reward = 0
            return self.board.flatten(), reward, self.game_over, {}
        else:
            return self.board.flatten(), -1, True, {}  # Invalid move

    def check_win(self):
        # Check rows, columns, and diagonals
        for i in range(3):
            if np.all(self.board[i, :] == self.current_player) or np.all(self.board[:, i] == self.current_player):
                return True
        if self.board[0, 0] == self.current_player and self.board[1, 1] == self.current_player and self.board[2, 2] == self.current_player:
            return True
        if self.board[0, 2] == self.current_player and self.board[1, 1] == self.current_player and self.board[2, 0] == self.current_player:
            return True
        return False


### DEFINE THE PLAYERS 
    
#Random Player
class RandomPlayer:
    def __init__(self, player):
        self.player = player

    def choose_action(self, board):
        available_actions = [i for i in range(9) if board[i] == 0]
        return random.choice(available_actions) if available_actions else None
    
#Random player who win in one action when he can
class SmartRandomPlayer:
    def __init__(self, player):
        self.player = player  # Player symbol: 1 for X, -1 for O

    def choose_action(self, board):
        # Convert the flat board to a 2D array for easier manipulation
        board_2d = np.array(board).reshape((3, 3))
        
        # Check for immediate win opportunities
        for action in range(9):
            row, col = divmod(action, 3)
            if board_2d[row, col] == 0:  
                board_2d[row, col] = self.player
                if self.check_win(board_2d, self.player):
                    return action  
                else:
                    board_2d[row, col] = 0  # Reset if not winning
                    
        # Default to random move if no immediate win is found
        available_actions = [i for i in range(9) if board[i] == 0]
        return random.choice(available_actions) if available_actions else None

    def check_win(self, board, player):
        # Check rows and columns
        for i in range(3):
            if all(board[i, :] == player) or all(board[:, i] == player):
                return True
        # Check diagonals
        if board[0, 0] == player and board[1, 1] == player and board[2, 2] == player:
            return True
        if board[0, 2] == player and board[1, 1] == player and board[2, 0] == player:
            return True
        return False

# The last opponent described in j) iii.
class StrategicPlayer:
    def __init__(self, player):
        self.player = player  # Player symbol: 1 for X, -1 for O

    def choose_action(self, board):
        board_2d = np.array(board).reshape((3, 3))
        
        # First, check if the player can win in the next move
        for action in range(9):
            row, col = divmod(action, 3)
            if board_2d[row, col] == 0:
                board_2d[row, col] = self.player
                if self.check_win(board_2d):
                    return action  
                board_2d[row, col] = 0 
                
        # Next, check if the opponent can win in their next move and block it
        opponent = -self.player
        for action in range(9):
            row, col = divmod(action, 3)
            if board_2d[row, col] == 0: 
                board_2d[row, col] = opponent
                if self.check_win(board_2d):
                    board_2d[row, col] = 0
                    return action  # Blocking move against the opponent
                board_2d[row, col] = 0  # Reset
                
        # If no immediate wins or blocks, choose a random move
        available_actions = [i for i in range(9) if board[i] == 0]
        return random.choice(available_actions) if available_actions else None

    def check_win(self, board_2d):
        # Check rows, columns, and diagonals for a win
        for i in range(3):
            if all(board_2d[i, :] == self.player) or all(board_2d[:, i] == self.player):
                return True
        if board_2d[0, 0] == self.player and board_2d[1, 1] == self.player and board_2d[2, 2] == self.player:
            return True
        if board_2d[0, 2] == self.player and board_2d[1, 1] == self.player and board_2d[2, 0] == self.player:
            return True
        return False

#The Learner
class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01):
        self.q_table = np.zeros((3**9, 9))  # 3^9 possible states and 9 actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate

    def state_to_index(self, state):
        #Converts a state into a unique index to be used in the Q-table
        index = 0
        for i in range(9):
            index += (3**i) * (state[i] + 1)
        return index

    def choose_action(self, state, available_actions):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(available_actions)
        else:
            state_index = self.state_to_index(state)
            q_values = self.q_table[state_index, available_actions]
            return available_actions[np.argmax(q_values)]

    def update_q_table(self, state, action, reward, next_state, done):
        state_index = self.state_to_index(state)
        next_state_index = self.state_to_index(next_state)
        future_rewards = np.max(self.q_table[next_state_index]) if not done else 0
        self.q_table[state_index, action] += self.learning_rate * (reward + self.discount_factor * future_rewards - self.q_table[state_index, action])
        if done:
            self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)


### TRAINING PHASE 
            

# Training the RL agent against the random player
def train_agent(episodes=1000):
    agent = QLearningAgent()
    game = TicTacToe()
    win_count = 0

    for episode in range(episodes):
        state = game.reset()
        done = False

        while not done:
            available_actions = [i for i in range(9) if state[i] == 0]
            action = agent.choose_action(state, available_actions)
            next_state, reward, done, _ = game.step(action)
            agent.update_q_table(state, action, reward, next_state, done)
            state = next_state

            if done:
                if game.winner == 1:
                    win_count += 1
                game.reset()

    return agent, win_count / episodes

# Train the agent and print out the win rate
q_agent, win_rate = train_agent(episodes=10000)
print(f"Win rate after training: {win_rate:.2f}")

def train_agent_with_smart_opponent(episodes=10000):
    agent = QLearningAgent()
    game = TicTacToe()
    win_count = 0
    opponent = SmartRandomPlayer(player=-1)  # Assuming the smart opponent plays as O

    for episode in range(episodes):
        state = game.reset()
        done = False
        while not done:
            if game.current_player == 1:  # Agent's turn
                available_actions = [i for i in range(9) if state[i] == 0]
                action = agent.choose_action(state, available_actions)
            else:  # Smart opponent's turn
                action = opponent.choose_action(state)
            
            next_state, reward, done, _ = game.step(action)
            if game.current_player == 1:  # Update Q-table only on the agent's turn
                agent.update_q_table(state, action, reward, next_state, done)
            state = next_state

            if done:
                if game.winner == 1:
                    win_count += 1
                game.reset()

    return agent, win_count / episodes

# Train the agent against the smart opponent and print out the win rate
q_agent, win_rate = train_agent_with_smart_opponent(episodes=10000) # If episode =100 then the q_agent plays worse than the SmartRP (win rate = 0.32)
print(f"Win rate against Smart Random Player after training: {win_rate:.2f}")

def train_agent_with_strategic_opponent(episodes=1000):
    agent = QLearningAgent()
    game = TicTacToe()
    win_count = 0
    opponent = StrategicPlayer(player=-1)

    for episode in range(episodes):
        state = game.reset()
        done = False

        while not done:
            if game.current_player == 1: 
                available_actions = [i for i in range(9) if state[i] == 0]
                action = agent.choose_action(state, available_actions)
            else: 
                action = opponent.choose_action(state)
            
            next_state, reward, done, _ = game.step(action)
            if game.current_player == 1:
                agent.update_q_table(state, action, reward, next_state, done)
            state = next_state

            if done:
                if game.winner == 1: 
                    win_count += 1
                game.reset()

    win_rate = win_count / episodes
    return agent, win_rate

# Train the agent against the strategic opponent and print out the win rate
q_agent, win_rate = train_agent_with_strategic_opponent(episodes=10000)
print(f"Win rate against Strategic Player after training: {win_rate:.2f}")
