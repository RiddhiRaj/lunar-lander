import gymnasium as gym
import neat
import numpy as np

# Define a function to evaluate a genome's fitness
def eval_genome(genome, config):
    # Create a neural network from the genome and config
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Create the Lunar Lander environment
    env = gym.make("LunarLander-v2")
    observation, _ = env.reset()
    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        # Normalize observation and pass it through the network
        action = np.argmax(net.activate(observation))
        
        # Take the action in the environment
        observation, reward, terminated, truncated, _ = env.step(action)
        
        # Accumulate rewards
        total_reward += reward

    # Add landing bonus
    if total_reward > 200:
        total_reward += 100
    # Add crash penalty
    elif total_reward < 0:
        total_reward -= 50

    return total_reward

# Define a function to evaluate the entire population
def eval_population(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

# Load NEAT configuration file
def run_neat():
    config_path = "config-feedforward.txt"
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

    # Run the NEAT algorithm for up to 'n' generations
    winner = population.run(eval_population, 400)

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
            action = np.argmax(net.activate(observation))
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
