🚀 Lunar Lander RL Agent Training
📌 Objective
Train a Reinforcement Learning agent to safely land a lunar module on the moon's surface using the OpenAI Gym LunarLander-v2 environment.
🗂️ Environment
Source: OpenAI Gym (LunarLander-v2)
State Space: 8 continuous values (position, velocity, angle, angular velocity, leg contact)
Action Space: 4 discrete actions (do nothing, fire left engine, fire main engine, fire right engine)
Reward: +100 to +140 for successful landing, -100 for crash, fuel penalty per timestep
Goal: Achieve an average reward of 200+ over 100 consecutive episodes
🔧 Methodology
Environment Setup — OpenAI Gym / Gymnasium with Box2D physics
Algorithm Selection — Deep Q-Network (DQN), Double DQN, Dueling DQN, or PPO
Agent Training — Experience replay, target network, gradient clipping
Evaluation — Average reward, landing success rate, fuel efficiency
📊 Results
Table
Algorithm	Avg Reward (100 eps)	Landing Success Rate
DQN	~180	~70%
Double DQN	~210	~85%
PPO	~250+	~95%
🚀 How to Run
bash
cd Lunar-Lander-RL/
pip install -r requirements.txt
python train_dqn.py
python train_ppo.py
# or open lunar_lander.ipynb
📁 Files
lunar_lander.ipynb — Full training and visualization notebook
train_dqn.py — DQN training script
train_ppo.py — PPO training script
agent/ — Saved trained agents
videos/ — Recorded landing episodes
results/ — Training curves and logs
requirements.txt — Dependencies
💡 Key Takeaways
PPO (Proximal Policy Optimization) significantly outperforms value-based methods on continuous control
Reward shaping and curriculum learning can accelerate convergence
Visualizing agent behavior helps debug policy failures
