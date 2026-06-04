from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from stable_baselines3 import DQN
import numpy as np
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
except Exception as e:
    print(f"Erreur lors du chargement: {e}")
    env = None
    model = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/symptoms", methods=["GET"])
def get_symptoms():
    if not env:
        return jsonify({"error": "Environnement non chargé"}), 500
    
    return jsonify({"symptoms": env.symptoms_list})

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
            
    # Capacités supposées à 0 (tous les services disponibles)
    cap_vec = np.zeros(env.n_services, dtype=np.float32)
    
    # Concaténation
    obs = np.concatenate([symptom_vec, cap_vec])
    
    # Prédiction
    action, _ = model.predict(obs, deterministic=True)
    
    predicted_service = env.get_service_name(int(action))
    
    return jsonify({
        "service": predicted_service,
        "action_id": int(action)
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
