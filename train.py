from algorithm.dqn import DQN
from farkle import Farkle
from tqdm import tqdm
# Définir les paramètres d'entraînement
nb_episodes = 5000
env = Farkle()
agent = DQN(nb_episodes, env)

# Entraîner l'agent
for episode in tqdm(range(nb_episodes), desc='Training...'):
    state = env.reset()
    done = False
    while not done:
        action = agent.action(state)
        next_state, reward, done = env.step_no_print(action)
        agent.souvenir(state, action, reward, next_state, done)
        state = next_state
    agent.replay()

# Sauvegarde du modèle
agent.qnetwork.save("outputs/farkle_model.h5")
