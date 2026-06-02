import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class HospitalEnv(gym.Env):
    def __init__(self, csv_path, max_capacity=10, max_steps=100):
        super(HospitalEnv, self).__init__()
        
        # Charger dataset
        self.df = pd.read_csv(csv_path)
        
        # Encoder les services
        self.encoder = LabelEncoder()
        self.df["service_label"] = self.encoder.fit_transform(self.df["Emergency_Service"])
        self.n_services = len(self.encoder.classes_)
        
        # Extraire tous les symptômes uniques
        self.symptom_cols = [f"Symptom_{i}" for i in range(1, 18) if f"Symptom_{i}" in self.df.columns]
        symptoms = set()
        for col in self.symptom_cols:
            unique_vals = self.df[col].dropna().unique()
            for val in unique_vals:
                if isinstance(val, str) and val.strip() != "":
                    symptoms.add(val.strip())
        self.symptoms_list = sorted(list(symptoms))
        self.n_symptoms = len(self.symptoms_list)
        
        # Capacités des services
        self.max_capacity = max_capacity
        self.current_capacities = np.zeros(self.n_services, dtype=np.int32)
        
        # Limite de l'épisode
        self.max_steps = max_steps
        self.current_step = 0
        
        # Actions possibles : choisir un des services
        self.action_space = spaces.Discrete(self.n_services)
        
        # Observation : vecteur de symptômes (binaire) + capacités actuelles
        obs_size = self.n_symptoms + self.n_services
        self.observation_space = spaces.Box(
            low=0.0,
            high=float(max(1.0, max_capacity)),
            shape=(obs_size,),
            dtype=np.float32
        )
        
        self.index = 0

    def _get_obs(self):
        # Vecteur de symptômes (Multi-Hot)
        symptom_vec = np.zeros(self.n_symptoms, dtype=np.float32)
        row = self.df.iloc[self.index]
        for col in self.symptom_cols:
            val = row[col]
            if isinstance(val, str) and val.strip() != "":
                idx = self.symptoms_list.index(val.strip())
                symptom_vec[idx] = 1.0
                
        # Vecteur de capacités
        cap_vec = self.current_capacities.astype(np.float32)
        
        # Concaténation
        return np.concatenate([symptom_vec, cap_vec])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_capacities = np.zeros(self.n_services, dtype=np.int32)
        self.current_step = 0
        
        self.index = np.random.randint(0, len(self.df))
        
        return self._get_obs(), {}

    def step(self, action):
        correct_action = self.df.loc[self.index, "service_label"]
        
        # Vérifier si le service est saturé
        is_saturated = self.current_capacities[action] >= self.max_capacity
        is_correct = (action == correct_action)
        
        # Calcul de la récompense
        if is_saturated:
            # Erreur Opérationnelle : envoyer un patient vers un service plein
            reward = -20.0
            # On rejette le patient dans ce service, la capacité n'augmente pas.
        elif not is_correct:
            # Erreur Médicale : mauvais service
            reward = -10.0
            self.current_capacities[action] += 1
        else:
            # Succès : bon service et non saturé
            reward = 10.0
            self.current_capacities[action] += 1
            
        # Simuler la sortie des patients (ex: chaque patient a 10% de chance de sortir à chaque étape)
        discharges = np.random.binomial(self.current_capacities, p=0.1)
        self.current_capacities -= discharges
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        truncated = False
        
        # Passer au patient suivant
        self.index = np.random.randint(0, len(self.df))
        
        return self._get_obs(), reward, done, truncated, {}

    def get_service_name(self, label):
        return self.encoder.inverse_transform([label])[0]
        
    def get_patient_symptoms(self):
        row = self.df.iloc[self.index]
        symptoms = []
        for col in self.symptom_cols:
            val = row[col]
            if isinstance(val, str) and val.strip() != "":
                symptoms.append(val.strip())
        return symptoms