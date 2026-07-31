CartPole-v1 DQN Agent
A clean, minimal implementation of Deep Q-Network (DQN) for the classic CartPole-v1 environment using Gymnasium, PyTorch, and NumPy.

Overview
This project trains a reinforcement learning agent to balance a pole on a cart using the DQN algorithm with the following techniques:
Experience Replay — stores and samples past transitions for stable learning
Target Network — a periodically updated copy of the policy network to reduce moving-target instability
Epsilon-Greedy Exploration — anneals from pure exploration to near-greedy action selection
Gradient Clipping — caps gradient norm at 10.0 for training stability
Huber Loss (Smooth L1) — robust loss function for Q-value regression

Requirements
| Package    | Version (tested) |
| ---------- | ---------------- |
| Python     | ≥ 3.8            |
| PyTorch    | ≥ 1.9            |
| Gymnasium  | ≥ 0.28           |
| NumPy      | ≥ 1.21           |
| Matplotlib | ≥ 3.3            |

Install dependencies:
bash
pip install torch gymnasium numpy matplotlib
Note: The code automatically uses CUDA if available; otherwise it falls back to CPU.
File Structure
plain
.
├── dqn_cartpole.py      # Main training script (this file)
└── README.md            # This file

Hyperparameters
| Parameter             | Value       | Description                                      |
| --------------------- | ----------- | ------------------------------------------------ |
| `ENV_NAME`            | CartPole-v1 | OpenAI Gymnasium environment identifier          |
| `NUM_EPISODES`        | 400         | Maximum training episodes                        |
| `GAMMA`               | 0.99        | Discount factor for future rewards               |
| `LR`                  | 1e-3        | Adam learning rate                               |
| `BATCH_SIZE`          | 64          | Mini-batch size for experience replay sampling   |
| `REPLAY_CAPACITY`     | 20,000      | Maximum size of the replay buffer                |
| `MIN_REPLAY_SIZE`     | 1,000       | Minimum transitions before optimization begins   |
| `TARGET_UPDATE_EVERY` | 10          | Episodes between target network synchronizations |
| `EPS_START`           | 1.0         | Initial exploration rate (fully random)          |
| `EPS_END`             | 0.02        | Final exploration rate                           |
| `EPS_DECAY_STEPS`     | 10,000      | Steps over which epsilon decays exponentially    |
| `SEED`                | 42          | Global random seed for reproducibility           |

How to Run
1. Train the Agent
bash
python dqn_cartpole.py
The script will:
Initialize the environment, networks, optimizer, and replay buffer.
Run the training loop, collecting experience and optimizing the Q-network.
Print progress every 20 episodes (reward, moving average, current epsilon).
Stop early if the agent achieves an average reward of ≥ 475 over the last 100 episodes (Gymnasium's "solved" threshold for CartPole-v1).
Save a training curve plot to cartpole_training_curve.png.
2. Evaluate the Trained Agent
After training, the script automatically runs watch_trained_agent() which evaluates the policy for 3 episodes in rgb_array mode (suitable for headless servers or Colab). You can modify the render_mode argument to "human" for a live window if running locally with a display.

Key Components
ReplayBuffer
A fixed-capacity deque that stores (state, action, reward, next_state, done) transitions. Provides uniform random sampling for mini-batch updates.
QNetwork
A simple fully-connected neural network:
plain
Input (state_dim) → Linear(128) → ReLU → Linear(128) → ReLU → Linear(action_dim)

select_action(...)
Implements epsilon-greedy policy. Chooses a random action with probability ε; otherwise selects the action with the highest predicted Q-value.
optimize(...)
Performs a single gradient descent step:
Computes current Q-values for taken actions.
Computes target Q-values using the target network and the Bellman equation.
Minimizes Smooth L1 loss between current and target Q-values.
Clips gradients to a maximum norm of 10.0.

Expected Results
With the default hyperparameters and seed 42, the agent typically solves CartPole-v1 (avg reward ≥ 475 over 100 consecutive episodes) within 150–250 episodes. The exact convergence speed may vary slightly depending on hardware and PyTorch version.

Sample Output
plain
Episode   20 | reward:   26.0 | avg(last 20):   22.4 | epsilon: 0.865
Episode   40 | reward:   33.0 | avg(last 20):   28.7 | epsilon: 0.741
...
Episode  180 | reward:  500.0 | avg(last 20):  489.2 | epsilon: 0.020
Solved at episode 183! Average reward (last 100): 475.3
Eval episode 1: reward = 500.0
Eval episode 2: reward = 500.0
Eval episode 3: reward = 500.0

Customization Tips
Faster convergence: Increase LR to 3e-3 or reduce TARGET_UPDATE_EVERY to 5.
More stable training: Increase REPLAY_CAPACITY to 50_000 or raise MIN_REPLAY_SIZE to 2_000.
Different environment: Change ENV_NAME and adjust state_dim / action_dim handling if the observation/action spaces differ from CartPole-v1.
Save model: Add torch.save(policy_net.state_dict(), "dqn_cartpole.pt") after training to persist weights.
Video recording: Replace watch_trained_agent() with a Gymnasium RecordVideo wrapper for saving MP4s.

License
This code is provided as-is for educational and research purposes. Feel free to modify and reuse.

References
Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature.
Gymnasium Documentation
PyTorch Documentation 
