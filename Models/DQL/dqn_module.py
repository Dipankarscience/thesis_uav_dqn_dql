import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_space, action_space):
        # Initialize the DQN architecture and hyperparameters
        self.state_space = state_space
        self.action_space = action_space
        self.memory = deque(maxlen=2000)  # Experience replay buffer
        self.gamma = 0.95  # Discount factor
        self.epsilon = 0.8  # Exploration rate (start)
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.85  # Decay rate for exploration
        self.learning_rate = 0.01  # Q-network learning rate
        self.model1 = self._build_model()  # First Q-network
        self.model2 = self._build_model()  # Second Q-network

    def _build_model(self):
        # Build the Q-network model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_space, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(len(self.action_space), activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(self.action_space)
        state = np.array(state)  # Convert the state to a NumPy array
        q_values = self.model1.predict(state.reshape(1, -1))[0] + self.model2.predict(state.reshape(1, -1))[0]
        return tuple(self.action_space[np.argmax(q_values)])

    def train(self, state, action, reward, next_state, done):
        # Train the DQN agent using experience replay
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) < 32:  # Start training only when there are enough samples in the buffer
            return

        minibatch = random.sample(self.memory, 32)
        minibatch_states = []
        minibatch_next_states = []

        for minibatch_state, minibatch_action, minibatch_reward, minibatch_next_state, minibatch_done in minibatch:
            minibatch_states.append(np.array(minibatch_state).reshape(2))  # Change to the actual size of your state
            minibatch_next_states.append(np.array(minibatch_next_state).reshape(2))  # Change to the actual size of your state

        minibatch_states = np.array(minibatch_states)
        minibatch_next_states = np.array(minibatch_next_states)

        # Alternate between two Q-networks for action selection and evaluation
        if np.random.rand() > 0.5:
            q_values = self.model1.predict(minibatch_states)
            next_q_values = self.model1.predict(minibatch_next_states)
            target_model = self.model2
        else:
            q_values = self.model2.predict(minibatch_states)
            next_q_values = self.model2.predict(minibatch_next_states)
            target_model = self.model1

        # Update the Q-values for the selected actions
        for i in range(len(minibatch)):
            state, action, reward, next_state, done = minibatch[i]
            target = reward
            if not done:
                next_state = np.array(next_state)
                target = reward + self.gamma * target_model.predict(next_state.reshape(1, -1))[0][np.argmax(next_q_values[i])]
            q_values[i][self.action_space.index(action)] = target

        # Train the model using the updated Q-values
        if np.random.rand() > 0.5:
            self.model1.fit(minibatch_states, q_values, verbose=0)
        else:
            self.model2.fit(minibatch_states, q_values, verbose=0)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay