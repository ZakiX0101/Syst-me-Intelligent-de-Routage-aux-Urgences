# Système Intelligent de Routage des Patients basé sur le Reinforcement Learning (RL)

Ce projet vise à optimiser l'orientation des patients vers les différents services hospitaliers en utilisant des algorithmes d'apprentissage par renforcement (Reinforcement Learning). Le système est conçu pour équilibrer simultanément l'exactitude des diagnostics médicaux et la gestion des capacités hospitalières en temps réel.

## Objectifs du Projet

L'agent RL (Reinforcement Learning) est capable de :
1. **Analyser les symptômes** des patients via un encodage Multi-Hot.
2. **Recommander le service hospitalier** le plus approprié en fonction de la pathologie.
3. **Tenir compte des capacités disponibles** de chaque service en temps réel pour éviter la saturation.
4. **Réduire la surcharge hospitalière** en pénalisant les assignations vers des services déjà pleins.

L'environnement simule des arrivées et des décharges (sorties) de patients pour fournir un flux continu réaliste, modélisé au travers de la librairie `Gymnasium`.

## Types d'Erreurs et Pénalités (Récompenses)

Le système de récompenses a été construit pour distinguer deux problèmes majeurs du triage hospitalier :
- **Succès (+10)** : Le patient est orienté vers le bon service, et le service a des lits disponibles.
- **Erreur Médicale (-10)** : Le patient est orienté vers le mauvais service (erreur de diagnostic).
- **Erreur Opérationnelle (-20)** : Le patient est orienté vers un service saturé (peu importe si le diagnostic est bon ou mauvais).

## Architecture du Projet

Le projet est organisé autour de la structure suivante :

```
Projet/
│
├── data/
│   └── DiseaseAndSymptoms_with_services.csv  # Jeu de données des maladies et symptômes associés aux services.
│
├── env/
│   ├── __init__.py
│   └── hospital_env.py                       # Environnement RL (Gymnasium) personnalisé avec gestion dynamique des capacités.
│
├── models/
│   └── hospital_routing_model.zip            # Modèle DQN entraîné sauvegardé.
│
├── test/
│   ├── evaluation.py                         # Script pour évaluer l'exactitude du modèle sur 1000 patients continus.
│   └── test_model.py                         # Script pour tester une prédiction individuelle avec affichage des symptômes.
│
├── train/
│   └── train_dqn.py                          # Script d'entraînement de l'agent DQN avec tracking.
│
└── requirements.txt                          # Dépendances Python nécessaires au projet.
```

## Modèle Utilisé (DQN)

Le modèle utilisé est un **Deep Q-Network (DQN)** fourni par la bibliothèque `stable-baselines3`.
- L'espace d'état (`observation_space`) est une combinaison continue/discrète : 
  - Un vecteur binaire (Multi-Hot Encoded) représentant l'absence ou la présence de chaque symptôme possible.
  - Un vecteur représentant le nombre de lits actuellement occupés dans chaque service.
- Le modèle a été entraîné sur **100 000 itérations (timesteps)** pour lui permettre d'apprendre des associations complexes entre les vecteurs de symptômes larges et les capacités disponibles.

## Installation

Assurez-vous de disposer de Python installé sur votre machine. Installez les dépendances nécessaires en exécutant la commande :

```bash
pip install -r requirements.txt
```

## Utilisation

### 1. Entraînement du modèle

Pour (ré)entraîner le modèle à partir de zéro, exécutez le script suivant :

```bash
python train/train_dqn.py
```
Le modèle entraîné sera sauvegardé dans le dossier `models/`.

### 2. Évaluation des Performances

Pour évaluer les performances globales de l'agent RL (taux de succès exact, erreurs médicales, erreurs opérationnelles), lancez :

```bash
python test/evaluation.py
```

### 3. Test Individuel

Pour observer la recommandation de l'agent sur un seul patient (avec affichage de ses symptômes, le service prédit vs réel et le statut du service), exécutez :

```bash
python test/test_model.py
```

## Interface Web

Une interface utilisateur web complète (Frontend en HTML/CSS/JS et Backend en Flask) a été développée pour permettre aux médecins et secrétaires de consulter facilement le modèle.

L'interface propose :
- **Recherche et sélection** faciles des symptômes à partir d'une liste interactive.
- **Thème Premium Dark/Light** avec des micro-animations ("glassmorphism").
- **Backend Flask** qui transforme les sélections en vecteur Multi-Hot et retourne la recommandation du modèle.

Pour lancer l'interface :
```bash
python app.py
```
Ouvrez ensuite `http://127.0.0.1:5000` dans votre navigateur.

<img width="1919" height="918" alt="image" src="https://github.com/user-attachments/assets/e321b5c3-2b2c-4ec6-a511-d207e718cf4c" />


## Performances du Modèle

Suite à l'évaluation du modèle sur 1000 patients continus (`python test/evaluation.py`), le système a atteint d'excellents résultats :

- **Accuracy (Précision Exacte)** : **97.60%**
- **Bons routages (Succès)** : 976
- **Erreurs Médicales** : 24
- **Erreurs Opérationnelles** : 0
- **Récompense moyenne par étape** : 9.52
- **Temps d'exécution** : ~0.27 secondes (très rapide pour le traitement en temps réel)
  <img width="634" height="226" alt="image" src="https://github.com/user-attachments/assets/245f125e-a803-4e85-b16d-7e5af198ff69" />


Ces performances montrent que l'agent RL a réussi à apprendre à router presque parfaitement les patients vers le bon service, tout en évitant complètement de saturer les capacités hospitalières.
