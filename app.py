from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from stable_baselines3 import DQN
import numpy as np
import torch
import os
import sys

# Ajouter le chemin du projet
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from env.hospital_env import HospitalEnv

app = Flask(__name__)
CORS(app)

# Dataset path
DATA_PATH = "data/DiseaseAndSymptoms_with_services.csv"
MODEL_PATH = "models/hospital_routing_model"

# Load environment and model
try:
    print("Chargement de l'environnement...")
    env = HospitalEnv(DATA_PATH)
    print("Chargement du modèle DQN...")
    model = DQN.load(MODEL_PATH)
    print("Modèle et environnement chargés avec succès.")
    
    # Variable globale pour suivre les capacités en temps réel
    real_capacities = np.zeros(env.n_services, dtype=np.int32)
except Exception as e:
    print(f"Erreur lors du chargement: {e}")
    env = None
    model = None
    real_capacities = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/symptoms", methods=["GET"])
def get_symptoms():
    if not env:
        return jsonify({"error": "Environnement non chargé"}), 500
    
    return jsonify({"symptoms": env.symptoms_list})

@app.route("/api/capacities", methods=["GET"])
def get_capacities():
    if not env:
        return jsonify({"error": "Environnement non chargé"}), 500
    
    capacities_data = []
    for i in range(env.n_services):
        capacities_data.append({
            "service": env.get_service_name(i),
            "current": int(real_capacities[i]),
            "max": int(env.max_capacity)
        })
        
    return jsonify({"capacities": capacities_data})

@app.route("/api/discharge", methods=["POST"])
def discharge():
    global real_capacities
    if not env:
        return jsonify({"error": "Environnement non chargé"}), 500
        
    # Simuler des sorties de patients (10% de chance pour chaque lit occupé)
    discharges = np.random.binomial(real_capacities, p=0.1)
    real_capacities -= discharges
    
    # S'assurer qu'on ne descend pas en dessous de 0 (normalement impossible avec binomial mais par sécurité)
    real_capacities = np.maximum(0, real_capacities)
    
    return jsonify({"message": "Lits libérés avec succès", "discharged": int(np.sum(discharges))})

@app.route("/api/predict", methods=["POST"])
def predict():
    if not env or not model:
        return jsonify({"error": "Modèle ou environnement non chargé"}), 500
        
    data = request.json
    selected_symptoms = data.get("symptoms", [])
    
    if not selected_symptoms:
        return jsonify({"error": "Aucun symptôme sélectionné"}), 400
        
    # Création du vecteur d'observation
    symptom_vec = np.zeros(env.n_symptoms, dtype=np.float32)
    for symptom in selected_symptoms:
        if symptom in env.symptoms_list:
            idx = env.symptoms_list.index(symptom)
            symptom_vec[idx] = 1.0
            
    # Utilisation des capacités réelles
    global real_capacities
    cap_vec = real_capacities.astype(np.float32)
    
    # Concaténation
    obs = np.concatenate([symptom_vec, cap_vec])
    
    # --- DEBUT DE L'ACTION MASKING ---
    # Convertir l'observation en tenseur PyTorch et la passer au réseau
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(model.device)
    with torch.no_grad():
        q_values = model.q_net(obs_tensor).cpu().numpy()[0]
        
    # Appliquer le masque pour les services saturés
    has_available_service = False
    for i in range(env.n_services):
        if real_capacities[i] >= env.max_capacity:
            q_values[i] = -np.inf
        else:
            has_available_service = True
            
    # S'il ne reste vraiment aucune place nulle part, on laisse le modèle choisir normalement
    if not has_available_service:
        # Reprendre l'ancienne prédiction
        action, _ = model.predict(obs, deterministic=True)
        action_idx = int(action)
    else:
        # Prendre l'action avec la plus grande Q-Value parmi celles disponibles
        action_idx = int(np.argmax(q_values))
    # --- FIN DE L'ACTION MASKING ---
    
    predicted_service = env.get_service_name(action_idx)
    
    # Vérifier si le service était déjà saturé avant d'y envoyer le patient
    was_saturated = real_capacities[action_idx] >= env.max_capacity
    
    # Mettre à jour la capacité réelle
    real_capacities[action_idx] += 1
    # On peut éventuellement borner à env.max_capacity si on rejette vraiment le patient
    # Mais ici on le laisse dépasser pour montrer qu'il est surchargé à l'interface
    
    return jsonify({
        "service": predicted_service,
        "action_id": action_idx,
        "is_saturated": bool(was_saturated),
        "capacity_after": int(real_capacities[action_idx])
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
