import psutil
from tqdm import tqdm
from farkle import Farkle
from algorithm.dqn import DQN
import os
import time
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


def train(load_model_path=None, nb_episodes=1000):
    nameModel = 'FarkleAI_DQN'
    env = Farkle()
    agent = DQN(nb_episodes=nb_episodes, env=env)

    if load_model_path:
        agent.load_agent(load_model_path)
        print(f"Modèle chargé depuis : '{load_model_path}'.")

    max_game_steps = 140  # Nombre maximum d'étapes par partie

    starttime = time.time()

    agent_win = 0
    random_win = 0
    draws = 0  # Compteur de matchs nuls
    total_rewards = []

    for episode in range(nb_episodes):
        s = env.reset()
        game_steps = 0
        episode_reward = 0
        done = False  # Assurez-vous que 'done' est défini avant la boucle

        while not env.game_over and game_steps < max_game_steps:
            # Tour de l'agent
            while True:
                # Vous pouvez commenter ou limiter les impressions pour économiser de la mémoire
                os.system('cls' if os.name == 'nt' else 'clear')
                print(
                    f"Épisode : {episode+1} | Étape : {game_steps+1} | epsilon : {agent.epsilon}")
                print(f"[A:{agent_win} | R:{random_win} | D:{draws}]")

                action = agent.action(s)

                stp1, r, done = env.step_w_print(action)

                # Enregistrement des états dans la mémoire
                agent.souvenir(s=s, a=action, r=r, stp1=stp1, done=done)
                agent.replay()  # Vous pouvez appeler replay ici ou après chaque épisode

                s = stp1
                game_steps += 1
                episode_reward += r
                if env.game_over or done:
                    print('Tour agent - terminé')
                    print(
                        f"Agent banks {env.env_state[1]} points. Total {env.env_state[0]}")
                    break

            if env.game_over:
                if env.env_state[0] >= env.pts_victory:
                    print('Partie terminée. Agent a gagné.')
                    agent_win += 1
                else:
                    print('Partie terminée.')
                    random_win += 1
                break  # Fin de l'épisode

            # Tour de l'adversaire
            s, opp_done = env.opponentv3_turn()
            game_steps += 1

            if env.game_over:
                if env.opponent_score >= env.pts_victory:
                    print('Partie terminée. L\'adversaire a gagné.')
                    random_win += 1
                else:
                    print('Partie terminée.')
                    agent_win += 1
                break  # Fin de l'épisode

            env.start_agent_turn()
            s = np.array(env.env_state.copy())

            print(env.dice_state)
            # time.sleep(3.5)

        # Déterminer le résultat de la partie
        if env.env_state[0] >= env.pts_victory:
            # Déjà compté dans agent_win
            pass
        elif env.opponent_score >= env.pts_victory:
            # Déjà compté dans random_win
            pass
        elif game_steps >= max_game_steps:
            print('Partie terminée. Limite de pas atteinte. Match nul.')
            draws += 1
        else:
            print('Partie terminée. Aucun gagnant.')
            draws += 1

        total_rewards.append(episode_reward)

        if episode % 50 == 0:
            agent.update_tnetwork()

        if episode % 200 == 0:
            model_path = f'models/{nameModel}_{episode}.h5'
            agent.qnetwork.save(model_path)

    model_path = f'models/{nameModel}_{episode}.h5'
    agent.qnetwork.save(model_path)

    print("Entraînement terminé et modèle sauvegardé.\nDonnées d'entraînement enregistrées.")
    print(f"Nombre total de victoires de l'agent : {agent_win}")
    print(f"Nombre total de victoires de l'adversaire : {random_win}")
    print(f"Nombre total de matchs nuls : {draws}")

    plt.figure(figsize=(10, 5))
    plt.plot(agent.loss_history)
    plt.xlabel('Updates.')
    plt.ylabel('Loss.')
    plt.title("Loss during training")
    plt.savefig('Loss_training.png')
    plt.close()

    window_size = 100
    if len(total_rewards) >= window_size:
        moving_avg_rewards = np.convolve(
            total_rewards, np.ones(window_size)/window_size, mode='valid')
    else:
        moving_avg_rewards = total_rewards

    # Tracer la courbe des récompenses moyennes
    plt.figure(figsize=(10, 5))
    plt.plot(moving_avg_rewards,
             label=f'moving averages (window={window_size})')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title(
        "Evolution of Average Rewards during training - DQN with priorized experience replay")
    plt.legend()
    plt.savefig('Average_rewards_training.png')
    plt.close()


def evaluate_model(model_path, nb_games):
    env = Farkle()
    agent = DQN(nb_episodes=1, env=env)
    agent.load_agent(model_path)
    agent.epsilon = 0
    agent_wins = 0
    victoire = 5000
    nb_games = nb_games
    i = 0

    dice_selection_counter = Counter()
    farkle_counts = []
    episode_scores = []
    steps_per_episode = []

    process = psutil.Process()

    for i in tqdm(range(nb_games), desc="Evaluating model"):
        s = np.array(env.reset())
        done = False
        farkle_count = 0
        game_steps = 0
        # affichage de la mémoire par curiosite :)
        os.system('cls' if os.name == 'nt' else 'clear')
        memory_mb = process.memory_info().rss/(1024**2)
        tqdm.write(f"Memory usage: {memory_mb:.2f} mb")

        while env.env_state[0] <= victoire or env.opponent_score <= victoire:
            while not done:
                action = agent.action(s)

                if action in range(6):
                    dice_selection_counter[action + 1] += 1

                stp1, r, done = env.step_w_print(action)

                if r == -10:
                    farkle_count += 1

                s = stp1

                if done:

                    break
                if env.env_state[0] >= victoire:

                    break
                game_steps += 1
            if env.env_state[0] >= victoire:
                # if env.env_state[0] >= victoire:
                agent_wins += 1
                break

            # not sure about this part
            s, oppdone = env.opponentv3_turn()
            s = s
            done = False
            # if env.game_over:
            if env.opponent_score >= victoire:

                break
            env.start_agent_turn()
            s = np.array(env.env_state.copy())

        i += 1
        steps_per_episode.append(game_steps)
        farkle_counts.append(farkle_count)
        episode_scores.append(env.env_state[0])

    win_rate = (agent_wins / nb_games)
    farkle_mean = sum(farkle_counts)/len(farkle_counts)
    steps_mean = sum(steps_per_episode) / len(steps_per_episode)
    score_mean = sum(episode_scores) / len(episode_scores)
    dice_faces = list(range(1, 7))
    frequencies = [dice_selection_counter[face] for face in dice_faces]

    plt.figure(figsize=(10, 5))
    plt.bar(dice_faces, frequencies, color='skyblue')
    plt.xlabel('Dices')
    plt.ylabel('Frequency of selection')
    plt.title('Frequency of chosen dice - Simple DDQN')
    plt.xticks(dice_faces)
    plt.savefig('frequency_selected_dice(simpleddqn).jpg')

    return win_rate, steps_mean, farkle_mean, score_mean


if __name__ == "__main__":
    model_path = 'outputs/farkle_model.h5'
    score_model, mean_step, mean_farkle, mean_score = evaluate_model(
        model_path=model_path, nb_games=10)
    print(
        f'Score model: {score_model}\nMean step: {mean_step}\nFarkle mean: {mean_farkle}\nMean score: {mean_score}')
