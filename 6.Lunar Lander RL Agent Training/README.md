 Lunar Lander RL - End-to-End Colab Deployment
A complete, self-contained Python script that trains a Deep Q-Network (DQN) agent on OpenAI Gym's LunarLander-v3 environment, builds an interactive Streamlit web UI, and deploys it to a public URL directly from Google Colab.
 Features
Table

| Feature                | Description                                                               |
| ---------------------- | ------------------------------------------------------------------------- |
| **Auto-Training**      | Trains a DQN agent from scratch (~5-10 min on GPU) or loads a saved model |
| **Interactive UI**     | Streamlit web app with sliders for episodes, steps, and seed              |
| **Live Visualization** | Reward charts, success/crash indicators, frame-by-frame replay            |
| **Public URL**         | Exposes the app via ngrok tunnel (shareable worldwide)                    |
| **Zero Config**        | Single-file, paste-and-run — no setup required                            |

 App Preview
The deployed web app includes:
Sidebar Controls: Adjust episodes (1-20), max steps (100-2000), random seed
Run Simulation: Execute the RL agent with live progress bar
Metrics Dashboard: Mean reward, best reward, success rate (%)
Reward Chart: Color-coded bar chart (green = landed, orange = partial, red = crashed)
Episode Replay: Frame slider to scrub through the first episode
Model Info: Expandable section showing DQN hyperparameters

 Quick Start
Option A: ngrok Public URL (Recommended)
Get a free ngrok token at dashboard.ngrok.com
Open Google Colab
Create a new notebook with GPU runtime (Runtime → Change runtime type → GPU)
Paste the entire colab_deploy_e2e.py code into one cell
Set your token in the code:
Python
NGROK_TOKEN = "2abc123..."  # <-- paste here
Run the cell (Ctrl+Enter)
Copy the public URL printed in the output — share it!

Option B: Colab Native Proxy (No Token)
If you don't want to sign up for ngrok:
Change this line in the code:
Python
TUNNEL = "colab"  # instead of "ngrok"
Run the cell
Copy the Colab proxy URL
 Limitation: URL only works while your Colab session is active
 
 File Structure
plain
colab_deploy_e2e.py      # Main script (paste into Colab)
├── Installs deps         # swig, gymnasium[box2d], streamlit, pyngrok, etc.
├── Trains DQN model      # Saved to /content/lunar_lander_model.zip
├── Writes app.py         # Streamlit UI file
└── Deploys tunnel        # ngrok or Colab native proxy

 Configuration
| Variable          | Default       | Description                                    |
| ----------------- | ------------- | ---------------------------------------------- |
| `FRAMEWORK`       | `"streamlit"` | Web framework (`"streamlit"` or `"flask"`)     |
| `TUNNEL`          | `"ngrok"`     | Tunnel provider (`"ngrok"` or `"colab"`)       |
| `PORT`            | `8501`        | Local server port (auto-set per framework)     |
| `NGROK_TOKEN`     | `""`          | Your ngrok authtoken (required for public URL) |
| `total_timesteps` | `200000`      | RL training steps (reduce for faster testing)  |


 Model Architecture
plain
Algorithm:     DQN (Deep Q-Network)
Policy:        MlpPolicy
Network:       [256, 256, 128] fully connected
Activation:    ReLU
Learning Rate: 1e-3
Buffer Size:   100,000
Batch Size:    128
Gamma:         0.99
Exploration:   1.0 → 0.05 over 30% of training

 Requirements
All dependencies are auto-installed by the script:
gymnasium[box2d] — Lunar Lander environment
stable-baselines3 — DQN implementation
streamlit — Web UI framework
pyngrok — Public tunneling
torch — Neural network backend
numpy, matplotlib — Data & plotting
System dependency (auto-installed via apt-get):
swig — Required to compile Box2D physics engine

 Troubleshooting
| Error                          | Cause                         | Fix                                               |
| ------------------------------ | ----------------------------- | ------------------------------------------------- |
| `Box2D not installed`          | Missing `swig` system package | Script now auto-runs `apt-get install swig`       |
| `ngrok: authentication failed` | Invalid or missing token      | Get free token at ngrok.com                       |
| `CUDA out of memory`           | GPU memory full               | Restart Colab runtime (Runtime → Restart session) |
| `ModuleNotFoundError`          | Packages not installed        | Re-run the first code block                       |
| `URL not accessible`           | ngrok free tier limit reached | Wait 1 minute or restart tunnel                   |

 Expected Results
After ~200k training steps, the DQN agent typically achieves:
| Metric       | Value    |
| ------------ | -------- |
| Mean Reward  | 150-250  |
| Success Rate | 60-90%   |
| Best Episode | 250-300+ |

 Useful Links
Get ngrok Authtoken
Stable-Baselines3 Docs
Gymnasium Lunar Lander
Streamlit Documentation

 License
MIT License — free to use, modify, and distribute.

 Credits
Built with:
Stable-Baselines3
Gymnasium
Streamlit
pyngrok
