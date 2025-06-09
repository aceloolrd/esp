import argparse
import os
import numpy as np
import gymnasium as gym
from esp import ESPPopulation

from utils import save_network, load_network
from network import RecurrentNetwork 

def train(args):
    env = gym.make('LunarLanderContinuous-v3', render_mode='human')

    pop = ESPPopulation(
        input_size=8,
        hidden_size=args.hidden_size,
        output_size=2,
        subpop_size=args.subpop_size,
        trials_per_individual=args.trials_per_individual if hasattr(args, 'trials_per_individual') else 10,
        alpha_cauchy=args.alpha_cauchy if hasattr(args, 'alpha_cauchy') else 1.0,
        stagnation_b=args.stagnation_b if hasattr(args, 'stagnation_b') else 20,
        mutation_rate=args.mutation_rate if hasattr(args, 'mutation_rate') else 0.1,
        crossover_rate=args.crossover_rate if hasattr(args, 'crossover_rate') else 0.5
    )

    reward_history = []
    loss_history = []

    os.makedirs(args.struct_dir, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\n=== Эпоха {epoch+1} ===")

        avg_fitness = pop.evaluate(env, n_episodes=args.episodes_per_eval, render=False)
        best_fitness_current = pop._compute_global_best_fitness_from_avg(avg_fitness)
        reward_history.append(best_fitness_current)
        loss_history.append(-best_fitness_current)

        print(f"Лучший avg_fitness на эпохе {epoch+1}: {best_fitness_current:.3f}")

        pop.best_history.append(best_fitness_current)
        if len(pop.best_history) == pop.stagnation_b:
            if best_fitness_current <= pop.best_history[0]:
                pop.burst_counter += 1
                pop.burst_mutation()
                if pop.burst_counter >= 2:
                    print("Две подряд burst-мутации без улучшения → адаптация структуры")
                    pop.adapt_structure(env, n_episodes=args.episodes_per_eval)
                    pop.burst_counter = 0
                    pop.best_history.clear()
                    avg_fitness = pop.evaluate(env, n_episodes=args.episodes_per_eval, render=False)
                    pop.select_and_breed(avg_fitness)
                    continue
                else:
                    pop.best_history.clear()
            else:
                pop.burst_counter = 0

        pop.select_and_breed(avg_fitness)

        net_vis = pop.get_best_network()
        visualize_network(net_vis, f"{args.struct_dir}/epoch_{epoch+1:04d}.png")

        if (epoch + 1) % 10 == 0:
            net_for_gif = pop.get_current_network()
            record_landing_gif(net_for_gif, epoch + 1)

    best_network = pop.get_best_network()
    save_network(best_network, args.save_weights)

    plot_metric(reward_history, "Best Avg Fitness", os.path.join(args.struct_dir, "reward_curve.png"))
    plot_metric(loss_history, "Loss (-Fitness)", os.path.join(args.struct_dir, "loss_curve.png"))

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESP for LunarLanderContinuous-v3")
    parser.add_argument("--train", action="store_true", help="Train ESP")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=12)
    parser.add_argument("--subpop_size", type=int, default=20)
    parser.add_argument("--episodes_per_eval", type=int, default=1)
    parser.add_argument("--struct_dir", type=str, default="structures")
    parser.add_argument("--save_weights", type=str, default="model.pkl")
    parser.add_argument("--load_weights", type=str, default="model.pkl")
    parser.add_argument("--test", action="store_true", help="Test ESP")
    parser.add_argument("--test_episodes", type=int, default=5)
    parser.add_argument("--visualize_structure", action="store_true")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--outfile", type=str, default="network.png")
    args = parser.parse_args()

    if args.train:
        train(args)
    elif args.test:
        test(args)
    elif args.visualize_structure:
        visualize(args)
    else:
        print("No mode specified. Use --train, --test or --visualize_structure.")