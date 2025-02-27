from collections import deque
import random
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
from collections import deque
import tensorflow as tf


def create_Qneural_network():
    model = models.Sequential([tf.keras.layers.Dense(64, input_shape=(
        13,), activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(9, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001), loss=tf.keras.losses.Huber())  # testing huberloss

    return model


def create_target_network():
    return create_Qneural_network()


class DQN():
    def __init__(self, nb_episodes, env) -> None:
        self.memoire = deque(maxlen=2000)
        self.priorities = deque(maxlen=500)  # Stocker les priorités
        self.gamma = 0.95
        self.N = nb_episodes
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = (
            self.epsilon_min / self.epsilon)**(1/self.N)
        self.loss_history = []

        self.batch_size = 8
        self.alpha = 0.6  # Hyperparamètre pour le PER
        self.beta = 0.4  # Hyperparamètre pour les poids d'importance
        self.beta_increment_per_sampling = 0.001  # Incrémentation de beta
        self.epsilon_priority = 1e-5  # Petit offset pour les priorités
        self.env = env
        self.train_start = 1000
        self.qnetwork = create_Qneural_network()
        self.tnetwork = create_target_network()
        self.update_tnetwork()

    def souvenir(self, s, a, r, stp1, done):
        self.memoire.append((s, a, r, stp1, done))

        # Calculer la priorité initiale (erreur max pour forcer un échantillonnage)
        # Priorité maximale initiale
        priority = max(self.priorities, default=1.0)
        self.priorities.append(float(priority))

    def load_agent(self, model_path):
        self.qnetwork = load_model(model_path)
        self.tnetwork = create_target_network()

    def get_valid_actions(self, s):
        valid_actions = []
        phase = s[11]
        dice_remaining = int(s[2])  # Nombre de dés restants à lancer
        dice_faces = s[3:9]  # Disponibilité des faces de dés (s4,1 à s4,6)
        has_selected_dice = s[12]

        if phase == 0:
            # Phase de sélection
            # Actions 0 à 5 : sélectionner un dé de face 1 à 6
            for i in range(6):
                die_face = i + 1
                die_count = dice_faces[i]
                if die_count > 0 and self.is_valid_selection(die_face, s):
                    valid_actions.append(i)

            if has_selected_dice == 1:
                valid_actions.append(6)

        elif phase == 1:
            if s[1] > 0:
                valid_actions.append(8)
            # Phase de décision
            # Actions 7 : relancer les dés restants
            # Actions 8 : banker (encaisser les points)
            valid_actions.append(7)
        return valid_actions

    def is_valid_selection(self, die_face, s):
        # Vérifier si le dé peut être sélectionné
        # die_face : valeur du dé (1 à 6)
        # s : état actuel
        # Implémenter la logique de vérification de validité

        # Récupérer la liste des dés restants
        dice_counts = s[3:9]
        dice_state = []
        for i in range(6):
            dice_state.extend([i + 1] * int(dice_counts[i]))
        # Vérifier si le dé peut être sélectionné
        return self.env.check_valid(self.env.selected_dice, die_face, dice_state)

    def update_tnetwork(self):
        self.tnetwork.set_weights(self.qnetwork.get_weights())

    def action(self, s):
        valid_actions = self.get_valid_actions(s)
        if len(valid_actions) == 0:
            return 6  # si 0 action valide on arrete la selection
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        else:
            q_values = self.qnetwork.predict(s.reshape(1, -1))
            q_values[0][[a for a in range(
                9) if a not in valid_actions]] = -np.inf
            return np.argmax(q_values[0])

    def update_priorities(self, indices, errors):
        # Mettre à jour les priorités en fonction des erreurs TD
        for idx, error in zip(indices, errors):
            priority = (abs(error) + self.epsilon_priority) ** self.alpha
            self.priorities[idx] = priority

    def sample_experiences(self):

        priorities_list = [float(p) for p in self.priorities]
        priorities = np.array(priorities_list)

        # Calculer les probabilités d'échantillonnage à partir des priorités
        scaled_priorities = priorities ** self.alpha
        sampling_probs = scaled_priorities / sum(scaled_priorities)

        # Utiliser la taille de self.priorities pour l'échantillonnage
        indices = np.random.choice(
            len(self.priorities), size=self.batch_size, p=sampling_probs)

        # Échantillonner les expériences correspondantes
        minibatch = [self.memoire[idx] for idx in indices]

        # Calculer les poids d'importance-sampling
        total = len(self.priorities)
        weights = (total * sampling_probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalisation des poids

        # Incrémenter beta pour atténuer le biais introduit par PER
        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        return minibatch, indices, weights

    def replay(self):
        if len(self.memoire) < self.train_start:
            return

        # Échantillonner les expériences et obtenir les poids d'importance
        minibatch, indices, weights = self.sample_experiences()

        # Liste pour stocker les erreurs TD
        td_errors = []
        replay_losses = []

        for i, (s, a, r, stp1, done) in enumerate(minibatch):
            # Reshape s pour qu'il ait la forme (1, 13)
            s_input = s.reshape(1, -1)
            target = self.qnetwork.predict(s_input)
            if done:
                target[0][a] = r
            else:
                # Reshape stp1 également pour prediction
                stp1_input = stp1.reshape(1, -1)
                t = self.tnetwork.predict(stp1_input)[0]
                td_target = r + self.gamma * np.amax(t)
                td_errors.append(td_target - target[0][a])
                target[0][a] = td_target

            history = self.qnetwork.fit(
                s_input, target, epochs=1, verbose=0,
                sample_weight=np.array([weights[i]])
            )
            loss_value = history.history['loss'][0]
            replay_losses.append(loss_value)

        # Mettre à jour les priorités dans la mémoire
        self.update_priorities(indices, td_errors)
        average_loss = np.mean(replay_losses)
        self.loss_history.append(average_loss)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
