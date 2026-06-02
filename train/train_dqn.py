from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from env.hospital_env import HospitalEnv
import os

# Dataset
DATA_PATH = "data/DiseaseAndSymptoms_with_services.csv"

# Environnement
env = HospitalEnv(DATA_PATH)
env = Monitor(env) # Pour tracker les récompenses

# Modèle DQN
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    target_update_interval=500,
    exploration_fraction=0.1,
    exploration_final_eps=0.05,
    verbose=1
)

# Callbacks
os.makedirs("models", exist_ok=True)
checkpoint_callback = CheckpointCallback(
    save_freq=10000, 
    save_path='./models/',
    name_prefix='rl_model'
)

# Entraînement avec beaucoup plus d'étapes car l'espace d'états est plus complexe
print("Début de l'entraînement...")
model.learn(
    total_timesteps=100000,
    callback=checkpoint_callback
)

# Sauvegarde
model.save("models/hospital_routing_model")
print("Modèle sauvegardé avec succès.")