import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from stable_baselines3 import DQN
from env.hospital_env import HospitalEnv

# Dataset
DATA_PATH = "data/DiseaseAndSymptoms_with_services.csv"

# Charger environnement
env = HospitalEnv(DATA_PATH)

# Charger modèle
model = DQN.load("models/hospital_routing_model")

# Test
obs, _ = env.reset()
symptoms = env.get_patient_symptoms()

action, _ = model.predict(obs, deterministic=True)

predicted_service = env.get_service_name(int(action))
real_service = env.get_service_name(env.df.loc[env.index, "service_label"])
is_saturated = env.current_capacities[int(action)] >= env.max_capacity

print("\n===== TEST INDIVIDUEL =====")
print("Symptômes du patient :", ", ".join(symptoms))
print("Service Prédit       :", predicted_service)
print("Service Réel         :", real_service)
if is_saturated:
    print("Statut du Service    : SATURE (Erreur Operationnelle)")
elif int(action) != env.df.loc[env.index, "service_label"]:
    print("Statut du Diagnostic : ERREUR MEDICALE")
else:
    print("Statut du Routage    : SUCCES")