End-to-End Render & Deployment
A complete end-to-end pipeline that trains a Deep Q-Network (DQN) agent on the LunarLander-v3 environment using Stable-Baselines3, then deploys an interactive Streamlit web app via tunnel (ngrok or Colab native proxy).
Designed for Google Colab — paste into a single cell and run.

Table of Contents
Overview
Features
Architecture
Prerequisites
Quick Start
Deployment Options
Project Structure
Configuration
Troubleshooting
Tech Stack
License

 Overview
This project demonstrates a full ML lifecycle:

| Stage          | What Happens                                                  |
| -------------- | ------------------------------------------------------------- |
| **1. Setup**   | Install system deps (swig, build-essential) + Python packages |
| **2. Train**   | Train a DQN agent on LunarLander-v3 (~5–10 min on GPU)        |
| **3. Persist** | Save the trained model to disk                                |
| **4. Serve**   | Launch a Streamlit app with interactive controls              |
| **5. Deploy**  | Expose the app publicly via ngrok or Colab proxy              |
The deployed app lets anyone run the trained RL agent, visualize episode replays, and inspect reward statistics — all from a browser.

 Features
 Pre-trained DQN Agent — Stable-Baselines3 with optimized hyperparameters
 Interactive Controls — Adjustable episodes, max steps, and seed via sidebar
 Live Metrics — Mean reward, best reward, and success rate
 Episode Replay — Frame-by-frame visualization of the first episode
 Reward Chart — Color-coded bar chart (green = success, red = crash)
 Public URL — One-click sharing via ngrok or Colab proxy
 Model Caching — Model auto-saved and reused across sessions
 
 Architecture
plain
┌─────────────────────────────────────────────────────────────┐
│                    GOOGLE COLAB RUNTIME                      │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐   │
│  │ System Deps │───▶│ Train DQN    │───▶│ Save Model    │   │
│  │ (swig, etc) │    │ (SB3 + GPU)  │    │ (.zip)        │   │
│  └─────────────┘    └──────────────┘    └───────────────┘   │
│                              │                              │
│                              ▼                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              STREAMLIT WEB APP (app.py)                │   │
│  │  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌────────┐ │   │
│  │  │ Controls│  │ Simulation│  │ Metrics │  │ Replay │ │   │
│  │  │ Sidebar │  │   Loop   │  │  Cards  │  │ Slider │ │   │
│  │  └─────────┘  └──────────┘  └─────────┘  └────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
│                              │                              │
│                              ▼                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              TUNNEL (ngrok / Colab Proxy)            │   │
│  │              Public HTTPS URL  ──────▶ Browser      │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

 Prerequisites
| Requirement     | Details                                                                                                        |
| --------------- | -------------------------------------------------------------------------------------------------------------- |
| **Platform**    | Google Colab (free GPU recommended)                                                                            |
| **Python**      | 3.10+                                                                                                          |
| **GPU**         | Optional but strongly recommended for training speed                                                           |
| **ngrok Token** | Free account at [dashboard.ngrok.com](https://dashboard.ngrok.com) *(optional — Colab proxy works without it)* |

 Quick Start
Step 1: Open Google Colab
Go to colab.research.google.com and create a new notebook.
Step 2: Paste the Code
Copy the entire script into one code cell.
Step 3: Configure (Optional)
Find this line and add your ngrok token for a persistent public URL:
Python
NGROK_TOKEN = ""   # <-- PASTE YOUR NGROK AUTHTOKEN HERE
 Tip: Leave it empty to use Colab's native proxy (session-only link).
Step 4: Run
Press Ctrl+Enter or click the play button. The script will:
Install all dependencies
Train (or load) the RL model
Write app.py
Launch Streamlit
Print a public URL
Step 5: Open the URL
Click the printed URL to interact with the Lunar Lander agent in your browser!

 Deployment Options
Option A: ngrok (Recommended for Sharing)
Provides a persistent public HTTPS URL that works even after Colab disconnects (as long as the runtime is active).
Python
TUNNEL = "ngrok"
NGROK_TOKEN = "your_token_here"   # Get free at dashboard.ngrok.com
Pros:
Clean public URL
Works on any device
No browser restrictions
Cons:
Requires free ngrok account
URL changes on new runtime
Option B: Colab Native Proxy (Zero Config)
Uses Google Colab's built-in port forwarding.
Python
TUNNEL = "colab"
NGROK_TOKEN = ""   # Ignored
Pros:
No account needed
Zero configuration
Cons:
URL is session-only (dies when runtime disconnects)
May have browser compatibility issues

 Project Structure
plain
/content/                          # Colab working directory
├── app.py                         # Streamlit application code
├── lunar_lander_model.zip         # Saved DQN model (auto-created)
└── [runtime logs & temp files]

File Descriptions
| File                     | Purpose                                                   |
| ------------------------ | --------------------------------------------------------- |
| `app.py`                 | Streamlit web app with simulation UI, metrics, and replay |
| `lunar_lander_model.zip` | Serialized DQN policy (PyTorch weights + config)          |

 Configuration
| Parameter               | Value          | Description                            |
| ----------------------- | -------------- | -------------------------------------- |
| `learning_rate`         | `1e-3`         | Adam optimizer learning rate           |
| `buffer_size`           | `100,000`      | Replay buffer capacity                 |
| `batch_size`            | `128`          | Mini-batch size for updates            |
| `gamma`                 | `0.99`         | Discount factor for future rewards     |
| `exploration_fraction`  | `0.3`          | Fraction of training for epsilon decay |
| `exploration_final_eps` | `0.05`         | Final epsilon (min exploration)        |
| `total_timesteps`       | `200,000`      | Total environment steps for training   |
| `device`                | `cuda` / `cpu` | Auto-detected GPU preference           |

App Controls (Runtime)
| Control   | Range       | Default | Description                     |
| --------- | ----------- | ------- | ------------------------------- |
| Episodes  | 1 – 20      | 5       | Number of episodes to simulate  |
| Max Steps | 100 – 2,000 | 1,000   | Step limit per episode          |
| Seed      | 0 – 1,000   | 42      | Random seed for reproducibility |

 Troubleshooting
"Box2D build failed" or similar compilation errors
Cause: swig or build-essential not installed before gymnasium[box2d].
Fix: The script already handles this by installing system deps first. If you see this error, restart the runtime and re-run.
"No module named 'gymnasium'"
Cause: Package installation incomplete or wrong order.
Fix: Re-run the cell. The script installs in correct order: base packages → gymnasium[box2d] → streamlit.
ngrok URL not working
Cause: Invalid or missing auth token, or ngrok rate limit reached.
Fix:
Verify token at dashboard.ngrok.com
Switch to TUNNEL = "colab" as fallback
Restart runtime if ngrok is stuck
Streamlit port already in use
Cause: Previous Streamlit process still running.
Fix: Restart the Colab runtime: Runtime → Restart runtime.
Model training is very slow
Cause: Running on CPU instead of GPU.
Fix: Go to Runtime → Change runtime type → Hardware accelerator → GPU.

 Tech Stack
| Layer             | Technology                                                     |
| ----------------- | -------------------------------------------------------------- |
| **RL Framework**  | [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) |
| **Environment**   | [Gymnasium](https://gymnasium.farama.org/) (LunarLander-v3)    |
| **Deep Learning** | [PyTorch](https://pytorch.org/)                                |
| **Web Framework** | [Streamlit](https://streamlit.io/)                             |
| **Tunneling**     | [pyngrok](https://pyngrok.readthedocs.io/) / Colab Proxy       |
| **Visualization** | Matplotlib                                                     |
| **Platform**      | Google Colab                                                   |

Model Details
plain
Architecture:    DQN (Deep Q-Network)
Policy Network:  MlpPolicy — [256, 256, 128]
Environment:     LunarLander-v3 (Box2D continuous control)
Observation:     8-dimensional state vector
Action Space:    4 discrete actions (do nothing, fire left, fire main, fire right)
Success Criteria: Reward > 200
Training Performance
Typical training time: 5–10 minutes on T4 GPU
Expected mean reward: 200+ (successful landings)
Model size: ~2 MB (compressed .zip)

 Future Improvements
[ ] Add video export (MP4/GIF) of episodes
[ ] Support for other RL algorithms (PPO, A2C, SAC)
[ ] Real-time training progress visualization
[ ] Docker containerization for local deployment
[ ] Deploy to Render / Heroku / AWS for permanent hosting
[ ] Add hyperparameter tuning UI
[ ] Multi-environment parallel training

 Contributing
This is a self-contained Colab script. To modify:
Fork the notebook
Adjust hyperparameters in the training section
Customize the Streamlit UI in the app_code string
Share your improved version!

 License
This project is open-source and available under the MIT License.

 Acknowledgments
Farama Foundation for Gymnasium
Stable-Baselines3 team
Streamlit for the amazing web framework
OpenAI for the original Lunar Lander environment
<p align="center">
  <b> Happy Landing! 🌙</b><br>
  <i>Built with Stable-Baselines3 + Streamlit + Google Colab</i>
</p>
