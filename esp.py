import numpy as np
import gymnasium as gym
from network import RecurrentNetwork
from collections import deque

class ESPPopulation:
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 subpop_size: int = 20,
                 trials_per_individual: int = 10,
                 alpha_cauchy: float = 1.0,
                 stagnation_b: int = 20,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.subpop_size = subpop_size
        self.trials_per_individual = trials_per_individual
        self.alpha_cauchy = alpha_cauchy
        self.stagnation_b = stagnation_b
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.subpopulations = [
            [self._random_individual() for _ in range(subpop_size)]
            for _ in range(hidden_size)
        ]

        self.cum_fitness = [
            np.zeros(subpop_size, dtype=np.float64)
            for _ in range(hidden_size)
        ]
        self.count_trials = [
            np.zeros(subpop_size, dtype=np.int32)
            for _ in range(hidden_size)
        ]

        self.best_history = deque(maxlen=stagnation_b)
        self.burst_counter = 0

    def _random_individual(self) -> np.ndarray:
        total_len = self.input_size + self.hidden_size + self.output_size
        return np.random.randn(total_len) * 0.1

    def assemble_network(self, hidden_indices: list[int]) -> RecurrentNetwork:
        w_ih = np.stack([
            self.subpopulations[i][hidden_indices[i]][:self.input_size]
            for i in range(self.hidden_size)
        ])
        w_hh = np.stack([
            self.subpopulations[i][hidden_indices[i]][self.input_size:self.input_size+self.hidden_size]
            for i in range(self.hidden_size)
        ])
        w_ho = np.stack([
            self.subpopulations[i][hidden_indices[i]][
                self.input_size+self.hidden_size:
                self.input_size+self.hidden_size+self.output_size]
            for i in range(self.hidden_size)
        ]).T
        return RecurrentNetwork(
            self.input_size,
            self.hidden_size,
            self.output_size,
            w_ih,
            w_hh,
            w_ho
        )

    def evaluate(self, env: gym.Env, n_episodes: int = 1, render: bool = False):
        for i in range(self.hidden_size):
            self.cum_fitness[i].fill(0.0)
            self.count_trials[i].fill(0)

        for i in range(self.hidden_size):
            for j in range(self.subpop_size):
                for t in range(self.trials_per_individual):
                    hidden_indices = []
                    for k in range(self.hidden_size):
                        hidden_indices.append(j if k == i else np.random.randint(0, self.subpop_size))
                    network = self.assemble_network(hidden_indices)
                    total_rewards = []
                    for ep in range(n_episodes):
                        obs, _ = env.reset()
                        done = False
                        episode_reward = 0.0
                        hidden_state = None
                        while not done:
                            action, hidden_state = network.forward(obs, hidden_state)
                            obs, reward, terminated, truncated, _ = env.step(action)
                            episode_reward += reward
                            done = terminated or truncated
                            if render:
                                env.render()
                        total_rewards.append(episode_reward)
                    avg_reward = np.mean(total_rewards)
                    self.cum_fitness[i][j] += avg_reward
                    self.count_trials[i][j] += 1

        avg_fitness = []
        for i in range(self.hidden_size):
            avg = self.cum_fitness[i] / np.maximum(self.count_trials[i], 1)
            avg_fitness.append(avg)
        return avg_fitness

    def select_and_breed(self, avg_fitness: list[np.ndarray]):
        for i in range(self.hidden_size):
            subpop = self.subpopulations[i]
            fitness_i = avg_fitness[i]
            sorted_idxs = np.argsort(-fitness_i)
            subpop_sorted = [subpop[idx].copy() for idx in sorted_idxs]
            top_k = max(1, self.subpop_size // 4)
            parents = subpop_sorted[:top_k]
            children = []
            for idx in range(0, top_k - 1, 2):
                if np.random.rand() < self.crossover_rate:
                    a, b = parents[idx], parents[idx + 1]
                    point = np.random.randint(1, len(a))
                    children.append(np.concatenate([a[:point], b[point:]]))
                    children.append(np.concatenate([b[:point], a[point:]]))
                else:
                    children.append(parents[idx].copy())
                    children.append(parents[idx + 1].copy())
            if top_k % 2 == 1:
                children.append(parents[-1].copy())
            m = len(children)
            keep_count = max(0, self.subpop_size - m)
            retained = subpop_sorted[:keep_count]
            subpop_new = retained + children
            half = self.subpop_size // 2
            for idx in range(half, self.subpop_size):
                perturb = self.alpha_cauchy * np.random.standard_cauchy(size=subpop_new[idx].shape)
                subpop_new[idx] += perturb
            for idx in range(self.subpop_size):
                if np.random.rand() < self.mutation_rate:
                    subpop_new[idx] += np.random.randn(*subpop_new[idx].shape) * 0.01
            self.subpopulations[i] = subpop_new

    def burst_mutation(self):
        print("=== BURST MUTATION ===")
        for i in range(self.hidden_size):
            avg_i = self.cum_fitness[i] / np.maximum(self.count_trials[i], 1)
            best_idx = int(np.argmax(avg_i))
            best_vector = self.subpopulations[i][best_idx]
            new_subpop = []
            for _ in range(self.subpop_size):
                perturb = self.alpha_cauchy * np.random.standard_cauchy(size=best_vector.shape)
                new_subpop.append(best_vector + perturb)
            self.subpopulations[i] = new_subpop
        for i in range(self.hidden_size):
            self.cum_fitness[i].fill(0.0)
            self.count_trials[i].fill(0)

    def adapt_structure(self, env: gym.Env, n_episodes: int = 1):
        print("=== ADAPT STRUCTURE ===")
        removed_any = True
        while removed_any:
            removed_any = False
            current_best = self._compute_global_best_fitness()
            old_subpops = [list(sp) for sp in self.subpopulations]
            old_hidden = self.hidden_size
            for i in range(old_hidden):
                tmp_pops = [old_subpops[k] for k in range(old_hidden) if k != i]
                tmp_hidden = old_hidden - 1
                tmp = ESPPopulation(
                    self.input_size, tmp_hidden, self.output_size,
                    self.subpop_size, self.trials_per_individual,
                    self.alpha_cauchy, self.stagnation_b,
                    self.mutation_rate, self.crossover_rate
                )
                tmp.subpopulations = [[ind.copy() for ind in sp] for sp in tmp_pops]
                best_tmp = tmp._compute_global_best_fitness_from_avg(
                    tmp.evaluate(env, n_episodes=n_episodes)
                )
                if best_tmp > current_best:
                    print(f"Удаляем подпопуляцию {i}: {current_best:.3f} → {best_tmp:.3f}")
                    self.subpopulations = tmp.subpopulations
                    self.hidden_size = tmp_hidden
                    removed_any = True
                    break
        # Если не удалили, добавляем новую и расширяем старые геномы
        if not removed_any:
            old_h = self.hidden_size
            self.hidden_size += 1
            # добавляем новую подпопуляцию (с правильной длиной генома)
            self.subpopulations.append([
                np.random.randn(self.input_size + self.hidden_size + self.output_size) * 0.1
                for _ in range(self.subpop_size)
            ])
            # расширяем все старые геномы
            for i in range(old_h):
                new_subpop = []
                for vec in self.subpopulations[i]:
                    ih = vec[:self.input_size]
                    hh = vec[self.input_size:self.input_size+old_h]
                    ho = vec[self.input_size+old_h:]
                    # расширяем hh и ho
                    new_hh = np.concatenate([hh, np.random.randn(1) * 0.1])
                    new_ho = np.concatenate([ho, np.random.randn(self.output_size) * 0.1])
                    new_vec = np.concatenate([ih, new_hh, new_ho])
                    new_subpop.append(new_vec)
                self.subpopulations[i] = new_subpop
        # сброс статистики
        self.cum_fitness = [np.zeros(self.subpop_size) for _ in range(self.hidden_size)]
        self.count_trials = [np.zeros(self.subpop_size, dtype=np.int32) for _ in range(self.hidden_size)]

    def _compute_global_best_fitness(self) -> float:
        best_value = -np.inf
        for i in range(self.hidden_size):
            avg_i = self.cum_fitness[i] / np.maximum(self.count_trials[i], 1)
            best_i = np.max(avg_i)
            if best_i > best_value:
                best_value = best_i
        return best_value

    def _compute_global_best_fitness_from_avg(self, avg_fitness: list[np.ndarray]) -> float:
        best = -np.inf
        for arr in avg_fitness:
            val = np.max(arr)
            if val > best:
                best = val
        return best

    def get_best_network(self) -> RecurrentNetwork:
        best_indices = []
        for i in range(self.hidden_size):
            avg_i = self.cum_fitness[i] / np.maximum(self.count_trials[i], 1)
            best_idx = int(np.argmax(avg_i))
            best_indices.append(best_idx)
        return self.assemble_network(best_indices)

    def get_current_network(self) -> RecurrentNetwork:
        return self.assemble_network([0] * self.hidden_size)