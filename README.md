# 🎲 FarkleAI

Un projet d'intelligence artificielle utilisant l'apprentissage par renforcement (Deep Q-Network) pour jouer au jeu de dés Farkle.

## 📋 Description

FarkleAI est une implémentation d'un agent d'intelligence artificielle qui apprend à jouer au jeu Farkle en utilisant l'algorithme Deep Q-Network (DQN) avec des améliorations avancées comme l'échantillonnage prioritaire d'expériences (Prioritized Experience Replay).

Le jeu Farkle est un jeu de dés où les joueurs tentent d'accumuler des points en sélectionnant des combinaisons de dés valides, tout en risquant de perdre leurs points s'ils continuent à jouer et obtiennent un "Farkle" (aucun dé valide).

## 🎯 Fonctionnalités

- **Agent DQN avancé** : Implémentation d'un réseau de neurones profond pour l'apprentissage par renforcement
- **Échantillonnage prioritaire** : Utilisation de Prioritized Experience Replay pour améliorer l'efficacité de l'apprentissage
- **Environnement de jeu complet** : Simulation complète du jeu Farkle avec toutes ses règles
- **Évaluation des performances** : Outils pour évaluer et analyser les performances de l'agent
- **Visualisation** : Graphiques de progression et d'analyse des stratégies

## 🏗️ Architecture du projet

```
FarkleAI-master/
├── algorithm/
│   └── dqn.py              # Implémentation de l'algorithme DQN
├── models/
│   └── FarkleAI_DQN_0.h5   # Modèle pré-entraîné
├── outputs/
│   └── farkle_model.h5     # Modèle sauvegardé
├── farkle.py               # Environnement de jeu Farkle
├── main.py                 # Script principal d'entraînement et d'évaluation
├── train.py                # Script d'entraînement simplifié
├── evaluate.py             # Script d'évaluation
└── requirement.txt         # Dépendances Python
```

## 🚀 Installation

1. **Cloner le repository**
   ```bash
   git clone https://github.com/eddys999/FarkleAI.git
   cd FarkleAI
   ```

2. **Installer les dépendances**
   ```bash
   pip install -r requirement.txt
   ```

## 🎮 Utilisation

### Entraînement de l'agent

```bash
# Entraînement simple
python train.py

# Entraînement avec paramètres personnalisés
python main.py
```

### Évaluation du modèle

```bash
# Évaluer un modèle pré-entraîné
python main.py
```

### Paramètres configurables

- `nb_episodes` : Nombre d'épisodes d'entraînement (défaut: 5000)
- `learning_rate` : Taux d'apprentissage (défaut: 0.0001)
- `epsilon` : Paramètre d'exploration (défaut: 1.0 → 0.1)
- `gamma` : Facteur de remise (défaut: 0.95)

## 🧠 Algorithme DQN

L'agent utilise un réseau de neurones profond avec les caractéristiques suivantes :

- **Architecture** : 3 couches denses (64, 512, 512 neurones)
- **Fonction d'activation** : ReLU
- **Optimiseur** : Adam avec taux d'apprentissage adaptatif
- **Fonction de perte** : Huber Loss
- **Mémoire** : Buffer de 2000 expériences avec échantillonnage prioritaire

### Améliorations avancées

- **Prioritized Experience Replay (PER)** : Les expériences importantes sont échantillonnées plus fréquemment
- **Target Network** : Réseau cible pour stabiliser l'apprentissage
- **Epsilon Decay** : Exploration décroissante au cours de l'entraînement

## 🎲 Règles du jeu Farkle

Le jeu Farkle implémenté suit ces règles :

- **Objectif** : Atteindre 5000 points
- **Dés** : 6 dés par tour
- **Combinaisons valides** :
  - 1 = 100 points
  - 5 = 50 points
  - Triplets = 100 × valeur du dé (1 = 1000)
  - Suite complète (1-6) = 1500 points
  - Trois paires = 1500 points
  - Deux triplets = 2500 points

- **Risque** : Si aucun dé valide n'est obtenu, le joueur perd tous les points du tour (Farkle)

## 📊 Métriques de performance

L'agent est évalué sur :

- **Taux de victoire** : Pourcentage de parties gagnées
- **Score moyen** : Score moyen par partie
- **Nombre de Farkles** : Fréquence des échecs
- **Nombre d'étapes** : Efficacité du jeu

## 🔧 Dépendances principales

- `tensorflow` : Framework de deep learning
- `numpy` : Calculs numériques
- `matplotlib` : Visualisation
- `tqdm` : Barres de progression
- `psutil` : Monitoring système

## 📈 Résultats

L'agent entraîné démontre :

- Une stratégie d'équilibre entre risque et récompense
- Une capacité d'apprentissage des combinaisons optimales
- Une adaptation aux différents états du jeu

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :

1. Fork le projet
2. Créer une branche pour votre fonctionnalité
3. Commiter vos changements
4. Pousser vers la branche
5. Ouvrir une Pull Request

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

*Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue sur GitHub !*
