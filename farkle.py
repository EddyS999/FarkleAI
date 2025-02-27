import numpy as np
import random
# from algorithm.dqn import DQN  <-- (Non utilisé directement ici)


class Farkle():
    def __init__(self):
        """
        Initialisation de l'environnement Farkle.
        On définit un état de taille 13 et on ajoute une variable turn_score
        pour accumuler le score du tour courant.
        """
        self.turn_number = 1
        self.current_player = 1
        self.dice_state = self.get_random_dice()

        self.env_state = [0] * 13  # état de taille 13
        self.pts_victory = 5000   # Score de victoire
        self.phase = 'selection'   # 'selection' ou 'decision'
        self.selected_dice = []    # dés sélectionnés lors du lancer courant
        self.has_selected_dice = False
        self.game_over = False
        self.opponent_score = 0

        # Nouveau : score cumulé du tour pour le joueur
        self.turn_score = 0

        self.env_state[9] = self.pts_victory - self.env_state[0]
        self.update_env_state()

    def reset(self):
        self.current_player = 1
        self.dice_state = self.get_random_dice()
        self.env_state = [0] * 13
        self.pts_victory = 5000
        self.phase = 'selection'
        self.selected_dice = []
        self.has_selected_dice = False
        self.game_over = False
        self.opponent_score = 0
        self.turn_score = 0  # réinitialise le score du tour
        self.env_state[9] = self.pts_victory - self.env_state[0]
        self.update_env_state()

        return np.array(self.env_state)

    def update_env_state(self):
        """
        Met à jour l'état de l'environnement.
        On renseigne notamment :
         - Le nombre de dés restants (index 2),
         - La distribution des faces (index 3 à 8),
         - Le score du tour courant (index 1) qui est la somme de turn_score
           et du potentiel score de la sélection en cours.
         - Un indicateur de phase (index 11) et de sélection (index 12).
        """
        self.env_state[2] = len(self.dice_state)

        counts = [0] * 6
        for die in self.dice_state:
            counts[die - 1] += 1
        for i in range(6):
            self.env_state[3 + i] = counts[i]

        self.env_state[9] = self.pts_victory - self.env_state[0]
        self.env_state[12] = int(self.has_selected_dice)
        self.env_state[11] = 0 if self.phase == 'selection' else 1

        # L'index 1 affiche le score cumulé du tour + le score potentiel de la sélection en cours
        potential = self.make_score(
            self.selected_dice.copy()) if self.selected_dice else 0
        self.env_state[1] = self.turn_score + potential

    def step_no_print(self, action):
        """
        Retourne (état, récompense, done) sans affichage.
        La logique de récompense est la suivante :
          - Aucun gain intermédiaire : seul le score banké (ou une pénalité en cas d'échec)
            est renvoyé en fin de tour.
          - L’agent accumule son score de tour dans turn_score.
        """
        reward = 0
        done = False

        # --- Phase de sélection ---
        if self.env_state[11] == 0:
            # Vérifier s'il existe des dés valides à sélectionner
            valid_dice = [
                die_face + 1
                for die_face in range(6)
                if self.env_state[3 + die_face] > 0 and
                self.check_valid(self.selected_dice,
                                 die_face + 1, self.dice_state)
            ]

            if len(valid_dice) == 0:
                # Aucun dé valide => Farkle
                if self.env_state[1] > 0:
                    # S'il y a déjà un score accumulé, on passe en phase décision (le joueur risque de perdre son tour s'il continue)
                    self.phase = 'decision'
                    self.env_state[11] = 1
                    reward = 0
                    done = False
                else:
                    # Pas de points : fin de tour avec pénalité
                    self.turn_score = 0
                    reward = -10
                    done = True

                self.update_env_state()
                return np.array(self.env_state), np.array(reward), np.array(done)

            # L'agent choisit un dé (actions 0 à 6)
            if action in range(0, 7):
                valid_actions = list(range(6))
                # L'action 6 signifie "terminer la sélection" si au moins un dé a déjà été choisi
                if len(self.selected_dice) > 0:
                    valid_actions.append(6)
                if action in valid_actions:
                    if action in range(0, 6):
                        die_face = action + 1
                        if self.env_state[3 + die_face - 1] > 0:
                            if self.check_valid(self.selected_dice, die_face, self.dice_state):
                                # Sélectionner le dé choisi
                                self.dice_state.remove(die_face)
                                self.selected_dice.append(die_face)
                                self.has_selected_dice = True
                                self.update_env_state()
                                reward = 0
                                done = False
                            else:
                                reward = 0
                                done = False
                        else:
                            reward = 0
                            done = False
                    elif action == 6 and self.selected_dice:
                        # L'agent termine la sélection du lancer courant : on ajoute le score de la sélection à turn_score
                        added = self.make_score(self.selected_dice.copy())
                        self.turn_score += added
                        # Réinitialiser la sélection courante
                        self.selected_dice = []
                        self.has_selected_dice = False
                        # Passage en phase de décision
                        self.phase = 'decision'
                        self.env_state[11] = 1
                        self.update_env_state()
                        reward = 0
                        done = False
                else:
                    reward = 0
                    done = False
            else:
                reward = 0
                done = False

        # --- Phase de décision ---
        elif self.env_state[11] == 1:
            if action == 7:
                # Continuer le tour (ne banke pas) : le score du tour (turn_score) est conservé
                if self.env_state[2] == 0:
                    # "Hot dice" : relancer 6 dés
                    self.dice_state = self.get_random_dice(6)
                    self.env_state[10] = 1  # Indicateur "Hot dice"
                else:
                    self.dice_state = self.get_random_dice(
                        int(self.env_state[2]))
                # Réinitialiser la sélection courante, mais garder turn_score
                self.selected_dice = []
                self.has_selected_dice = False
                self.phase = 'selection'
                self.env_state[11] = 0
                self.update_env_state()
                if self.is_farkle(self.dice_state):
                    # Si Farkle, le joueur perd tout le score du tour
                    self.turn_score = 0
                    self.update_env_state()
                    reward = -10
                    done = True
                else:
                    reward = 0
                    done = False

            elif action == 8:
                # Banker les points : le tour se termine et turn_score est ajouté au score total
                if self.turn_score == 0:
                    reward = -10  # pénalité si aucune point n'a été accumulé
                    done = True
                else:
                    reward = self.turn_score
                    self.env_state[0] += self.turn_score
                    # Réinitialiser le score du tour
                    self.turn_score = 0
                    self.update_env_state()
                    done = True
                    if self.env_state[0] >= self.pts_victory:
                        reward = self.env_state[0]
                        done = True
                        self.game_over = True
            else:
                reward = 0
                done = False

        else:
            reward = 0
            done = False

        return np.array(self.env_state), np.array(reward), np.array(done)

    def step_w_print(self, action):
        """
        Même logique que step_no_print mais avec quelques impressions pour le débogage.
        Le système de récompense est :
          - Récompense égale aux points bankés lors d'un banking réussi.
          - Pénalité en cas de Farkle ou d'une tentative de banker sans points.
        """
        reward = 0
        done = False

        if self.env_state[11] == 0:
            valid_dice = [die_face + 1 for die_face in range(6)
                          if self.env_state[3 + die_face] > 0 and
                          self.check_valid(self.selected_dice, die_face + 1, self.dice_state)]
            if len(valid_dice) == 0:
                print('No valid dice to select, Farkle.')
                print("Env state:", self.env_state)
                print("Dice state:", self.dice_state)
                print("Selected dice:", self.selected_dice)
                if self.env_state[1] > 0:
                    self.phase = 'decision'
                    self.env_state[11] = 1
                    reward = 0
                    done = False
                else:
                    self.turn_score = 0
                    reward = -10
                    done = True
                self.update_env_state()
                return np.array(self.env_state), np.array(reward), np.array(done)

            if action in range(0, 7):
                valid_actions = list(range(6))
                if len(self.selected_dice) > 0:
                    valid_actions.append(6)
                if action in valid_actions:
                    if action in range(0, 6):
                        die_face = action + 1
                        if self.env_state[3 + die_face - 1] > 0:
                            if self.check_valid(self.selected_dice, die_face, self.dice_state):
                                print('Agent selects die:', die_face)
                                self.dice_state.remove(die_face)
                                self.selected_dice.append(die_face)
                                self.has_selected_dice = True
                                self.update_env_state()
                                reward = 0
                                done = False
                            else:
                                reward = 0
                                done = False
                        else:
                            reward = 0
                            done = False
                    elif action == 6 and self.selected_dice:
                        # Fin de la sélection : on accumule le score du lancer courant
                        added = self.make_score(self.selected_dice.copy())
                        self.turn_score += added
                        self.selected_dice = []
                        self.has_selected_dice = False
                        self.phase = 'decision'
                        self.env_state[11] = 1
                        self.update_env_state()
                        reward = 0
                        done = False
                else:
                    reward = 0
                    done = False
            else:
                reward = 0
                done = False

        elif self.env_state[11] == 1:
            if action == 7:
                print('Agent continues.')
                if self.env_state[2] == 0:
                    self.dice_state = self.get_random_dice(6)
                    self.env_state[10] = 1
                else:
                    self.dice_state = self.get_random_dice(
                        int(self.env_state[2]))
                self.selected_dice = []
                self.has_selected_dice = False
                self.phase = 'selection'
                self.env_state[11] = 0
                self.update_env_state()
                if self.is_farkle(self.dice_state):
                    self.turn_score = 0
                    self.update_env_state()
                    reward = -10
                    done = True
                else:
                    reward = 0
                    done = False

            elif action == 8:
                if self.turn_score == 0:
                    print('Agent attempted to bank zero points. Penalized.')
                    reward = -10
                    done = True
                else:
                    reward = self.turn_score
                    self.env_state[0] += self.turn_score
                    self.turn_score = 0
                    self.update_env_state()
                    done = True
                    if self.env_state[0] >= self.pts_victory:
                        print('Agent won.')
                        reward = self.env_state[0]
                        done = True
                        self.game_over = True
            else:
                reward = 0
                done = False

        else:
            reward = 0
            done = False

        return np.array(self.env_state), np.array(reward), np.array(done)

    def start_agent_turn(self):
        self.current_player = 1
        self.dice_state = self.get_random_dice()
        self.selected_dice = []
        self.has_selected_dice = False
        self.phase = 'selection'
        self.turn_score = 0  # Réinitialiser le score du tour
        self.env_state[11] = 0  # Met à jour l'indicateur de phase
        self.update_env_state()

    def opponentv3_turn(self):
        """
        Tour de l'adversaire (joué aléatoirement).
        On accumule les points dans score_opponent_turn. En cas de banker, on ajoute ce score au total.
        """
        done = False
        self.current_player = 2
        self.score_opponent_turn = 0  # Nouveau score cumulé pour ce tour
        self.dice_state = self.get_random_dice()
        self.selected_dice = []
        self.has_selected_dice = False
        self.update_env_state()

        while True:
            if self.is_farkle(self.dice_state):
                # Farkle : l'adversaire perd tout ce tour
                self.score_opponent_turn = 0
                self.dice_state = []
                done = True
                break

            selectable_dice = [die for die in set(self.dice_state)
                               if self.check_valid([], die, self.dice_state)]
            if not selectable_dice:
                break

            selected_die = random.choice(selectable_dice)
            if selected_die in self.dice_state:
                self.dice_state.remove(selected_die)
                self.selected_dice.append(selected_die)
            else:
                break

            # Accumuler le score de la sélection courante
            added = self.make_score(self.selected_dice.copy())
            self.score_opponent_turn += added
            # Réinitialiser la sélection pour le prochain lancer
            self.selected_dice = []

            decision = random.choice(['bank', 'continue'])
            if decision == 'bank':
                self.opponent_score += self.score_opponent_turn
                self.dice_state = []
                break
            else:
                if not self.dice_state:
                    self.dice_state = self.get_random_dice(6)
                else:
                    self.dice_state = self.get_random_dice(
                        len(self.dice_state))
                continue

        if self.opponent_score >= self.pts_victory:
            done = True
            self.game_over = True

        self.current_player = 1
        self.turn_number += 1
        self.update_env_state()
        return np.array(self.env_state), np.array(done)

    def check_valid(self, deck, valeur, selection):
        """
        Vérifie si un dé de valeur donnée peut être sélectionné.
        Pour les 1 et 5, toujours valide.
        Sinon, il faut au moins trois dés identiques.
        """
        # total_count_in_roll = deck.count(valeur) + selection.count(valeur)
        # current_count_in_selected = deck.count(valeur)

        # if valeur == 1 or valeur == 5:
        #     return True
        # if total_count_in_roll >= 3 and current_count_in_selected < 3:
        #     return True

        total_count_in_roll = deck.count(valeur) + selection.count(valeur)
        if valeur == 1 or valeur == 5:
            return True

        # Pour les autres dés, la sélection est valide si le total dans le lancer est au moins 3
        if total_count_in_roll >= 3:
            return True

        return False

    def is_farkle(self, selection):
        """
        Un Farkle se produit si aucun dé de la sélection ne peut être joué.
        """
        return not any(self.check_valid([], valeur, selection) for valeur in selection)

    def make_score(self, deck):
        """
        Calcule le score en fonction des dés sélectionnés.
        - Suite complète, trois paires, deux triplets sont traités.
        - Pour les triplets et plus, les points sont doublés par dé en plus.
        - Les 1 et 5 individuels rapportent respectivement 100 et 50 points.
        """
        score = 0
        counts = {x: deck.count(x) for x in set(deck)}
        # Suite complète (1-6)
        if sorted(deck) == [1, 2, 3, 4, 5, 6]:
            return 1500
        # Trois paires
        if len(deck) == 6 and len(set(deck)) == 3 and all(count == 2 for count in counts.values()):
            return 1500
        # Deux triplets
        if len(deck) == 6 and len(set(deck)) == 2 and all(count == 3 for count in counts.values()):
            return 2500
        # Comptage des triplets et plus
        for num, count in counts.items():
            if count >= 3:
                if num == 1:
                    score += 1000 * (2 ** (count - 3))
                else:
                    score += num * 100 * (2 ** (count - 3))
        for num, count in counts.items():
            if num == 1 and count < 3:
                score += 100 * count
            elif num == 5 and count < 3:
                score += 50 * count

        return score

    def get_random_dice(self, num_dice=6):
        """
        Génère une liste de num_dice dés aléatoires.
        """
        return [random.randint(1, 6) for _ in range(num_dice)]
