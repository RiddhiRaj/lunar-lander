import gymnasium as gym
import neat
import numpy as np
import os

# Function to normalize the observation
def normalize_observation(observation):
    return (observation - np.min(observation)) / (np.max(observation) - np.min(observation))

# Define a function to evaluate a genome's fitness
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitnesses = []
    
    for _ in range(2):  # Evaluate 2 times
        env = gym.make("LunarLander-v2")
        observation, _ = env.reset()
        total_reward = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            normalized_observation = normalize_observation(observation)
            action = np.argmax(net.activate(normalized_observation))
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

        # Add landing bonus and crash penalty
        if total_reward > 200:
            total_reward += 100
        elif total_reward < 0:
            total_reward -= 50

        fitnesses.append(total_reward)
    
    return sum(fitnesses) / len(fitnesses)  # Return average fitness

# Define a function to evaluate the entire population
def eval_population(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

# Load NEAT configuration file
def run_neat():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # Create a population based on the NEAT configuration
    population = neat.Population(config)

    # Add reporters to show progress
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Add checkpointing
    checkpoint = neat.Checkpointer(generation_interval=100, time_interval_seconds=600)
    population.add_reporter(checkpoint)

    # Run the NEAT algorithm
    winner = population.run(eval_population, 500)

    print('\nBest genome:\n{!s}'.format(winner))

    return winner, config

# Visualize the winner agent playing the game
def run_winner(winner, config, num_episodes=20):
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    env = gym.make("LunarLander-v2", render_mode="human")

    total_rewards = []
    for episode in range(num_episodes):
        observation, _ = env.reset()
        total_reward = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            normalized_observation = normalize_observation(observation)
            action = np.argmax(net.activate(normalized_observation))
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Reward = {total_reward}")

    env.close()
    avg_reward = sum(total_rewards) / num_episodes
    print(f"Average reward over {num_episodes} episodes: {avg_reward}")

if __name__ == "__main__":
    winner, config = run_neat()
    run_winner(winner, config)