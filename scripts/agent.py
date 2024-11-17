import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from buffer import ReplayBuffer

# HYPERPARAMETERS
A_HIDDEN_DIM = 128
C_HIDDEN_DIM = 128
A_LEARNING_RATE = 1e-4
C_LEARNING_RATE = 1e-4
ALPHA = 0.2
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPS = 1e-8
GAMMA = 0.99
TAU = 0.001
BUFFER_SIZE = 10000
BATCH_SIZE = 128
ACTION_MIN = -torch.pi / 3
ACTION_MAX = torch.pi / 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim: int=A_HIDDEN_DIM,
                 eps: float=EPS, log_sig_min: float=LOG_SIG_MIN, log_sig_max: float=LOG_SIG_MAX,
                 action_min: float=ACTION_MIN, action_max: float=ACTION_MAX):
        super(Actor).__init__()
        
        # Store variables
        self.log_sig_min = log_sig_min
        self.log_sig_max = log_sig_max
        self.eps = eps
        
        # Define Actor Network
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.sigma = nn.Linear(hidden_dim, action_dim)
        
        # Define Activation function
        self.activation = nn.ReLU()
        
        # Scale actions
        self.action_scale = torch.FloatTensor(action_max - action_min).to(device)
        self.action_shift = torch.FloatTensor(action_min).to(device)
        
        
    def forward(self, state):
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        
        mean = self.mu(x)
        log_std = self.sigma(x)
        log_std = torch.clamp(log_std, min=self.log_sig_min, max=self.log_sig_max)
                        
        return mean, log_std
    
    # USE THIS FUNCTION TO GET THE ACTIONS
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Sample from distribution with reparameterization trick
        normal = torch.rand_like(mean)
        action = mean + std * normal
        # log_prob = -0.5*log(2pi) - log(std) - ((x - mean)^2 / 2(std^2))
        log_prob = -0.5*torch.log(torch.tensor(torch.pi * 2)) - torch.log(std) - ((action - mean)**2 / (2*(std**2) + self.eps))
        
        # Enforce bounds
        action = (torch.tanh(action) + 1.0) / 2
        action = action * self.action_scale + self.action_shift
        
        return action, log_prob, mean        
    

# TODO consider using two Q networks as done in homework
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim: int=C_HIDDEN_DIM):
        super(Critic).__init__()
        
        # Create Critic Network
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        
        # Define activation function
        self.activation = nn.ReLU()
        
        
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        
        return x
        
    

class AgentSAC(nn.Module):
    def __init__(self, state_dim, action_dim, 
                 alpha: float=ALPHA, tau: float=TAU, gamma: float=GAMMA,
                 actor_lr: float=A_LEARNING_RATE, critic_lr: float=C_LEARNING_RATE,
                 buffer_size: int=BUFFER_SIZE, batch_size: int=BATCH_SIZE):
        super(AgentSAC).__init__()
        
        # Save variables
        self.alpha = alpha
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        
        # Define Actor and Critic
        self.Actor = Actor(state_dim=state_dim, action_dim=action_dim)
        self.Critic = Critic(state_dim=state_dim, action_dim=action_dim)
        
        # Define Target Critic
        self.TargetCritic = Critic(state_dim=state_dim, action_dim=action_dim)
        self.TargetCritic.load_state_dict(self.Critic.state_dict())
        
        # Define Replay Buffer
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(capacity=buffer_size, batch_size=batch_size)
        
        # Create Optimizers
        self.actor_optim = optim.Adam(self.Actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.Critic.parameters(), lr=self.critic_lr)
        
    def optimize_actor(self, states):
        actions, log_probs, _ = self.Actor.sample(states)
        
        # calculate Q-values
        q_values = self.Critic(states, actions)
        entropy = self.alpha * log_probs
        policy_loss = -(q_values - entropy)
        
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()
        
    def optimize_critic(self, states, actions, rewards, next_states, dones):
        rewards = rewards.unsqueeze(-1)
        dones = dones.unsqueeze(-1)
        
        # sample next_actions and log_probs from Actor
        next_actions, next_log_probs, _ = self.Actor.sample(next_states)
        
        # compute q-values for next state-action pairs
        with torch.no_grad():
            next_q_values = self.TargetCritic(next_states, next_actions)
            target_q_values = rewards + self.gamma * (1 - dones) * (next_q_values - self.alpha * next_log_probs)
            
        # get current q-values
        current_q_values = self.Critic(states, actions)
        
        # compute critic loss (TD error)
        q_loss = F.mse_loss(current_q_values, target_q_values)
        
        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()