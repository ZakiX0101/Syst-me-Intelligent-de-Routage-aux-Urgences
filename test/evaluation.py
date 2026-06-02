import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from stable_baselines3 import DQN
from env.hospital_env import HospitalEnv

# Dataset
DATA_PATH = "data/DiseaseAndSymptoms_with_services.csv"

# Charger environnement avec un long épisode pour simuler 1000 patients d'affilée
env = HospitalEnv(DATA_PATH, max_steps=1000)

# Charger modèle
model = DQN.load("models/hospital_routing_model")

# Variables statistiques
n_tests = 1000

correct_predictions = 0
medical_errors = 0
operational_errors = 0

total_reward = 0

start_time = time.time()

obs, _ = env.reset()

# Evaluation
for _ in range(n_tests):
    action, _ = model.predict(obs, deterministic=True)
    real_action = env.df.loc[env.index, "service_label"]
    
    # Analyser avant d'effectuer l'action pour les stats
    is_saturated = env.current_capacities[int(action)] >= env.max_capacity
    is_correct = (int(action) == real_action)
    
    if is_saturated:
        operational_errors += 1
    elif not is_correct:
        medical_errors += 1
    else:
        correct_predictions += 1

    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    
    if done or truncated:
        obs, _ = env.reset()

# Temps final
end_time = time.time()

# Statistiques
accuracy = (correct_predictions / n_tests) * 100
average_reward = (total_reward / n_tests)
execution_time = (end_time - start_time)

# Affichage
print("\n===== EVALUATION =====")
print(f"Nombre de patients testés : {n_tests}")
print(f"Bons routages (Succès)    : {correct_predictions}")
print(f"Erreurs Médicales         : {medical_errors}")
print(f"Erreurs Opérationnelles   : {operational_errors}")
print(f"Accuracy Exacte           : {accuracy:.2f}%")
print(f"Reward moyenne par step   : {average_reward:.2f}")
print(f"Temps d'exécution         : {execution_time:.2f} sec")