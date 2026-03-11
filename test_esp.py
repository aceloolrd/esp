"""
Тесты для верификации исправлений алгоритма ESP (esp.py).

Каждый тест ПРОВАЛИТСЯ на текущем (багованном) коде и
ПРОЙДЁТ после применения фиксов из плана.

Запуск:
    pip install pytest
    pytest test_esp.py -v
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from esp import ESPPopulation


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_env(obs_size: int, reward: float = 10.0):
    """Мок-среда: reset() → нулевое наблюдение, step() → done=True за 1 шаг."""
    env = MagicMock()
    obs = np.zeros(obs_size)
    env.reset.return_value = (obs, {})
    env.step.return_value = (obs, reward, True, False, {})
    return env


def force_add_branch(pop: ESPPopulation, env) -> None:
    """
    Настраивает популяцию так, чтобы adapt_structure выбрала ветку
    «добавить нейрон»:
      • текущий current_best высокий → удаление ни одного нейрона не улучшает
      • env возвращает очень плохую награду для урезанных вариантов
    """
    for i in range(pop.hidden_size):
        pop.cum_fitness[i][:] = 9999.0
        pop.count_trials[i][:] = 1
    env.step.return_value = (np.zeros(pop.input_size), -9999.0, True, False, {})


# ─────────────────────────────────────────────────────────────────────────────
# БАГ 1: adapt_structure должна брать current_best из свежей оценки,
#         а не из обнулённых cum_fitness после burst_mutation
# ─────────────────────────────────────────────────────────────────────────────

class TestBug1CurrentBest:

    def test_no_spurious_removal_after_burst(self):
        """
        Сценарий: среда возвращает одинаковую награду (10.0) и для полной,
        и для урезанной популяции. Удаление «не лучше» → нейрон не должен удаляться.

        БАГ: после burst_mutation cum_fitness=0 → current_best=0 → best_tmp(10) > 0
             → нейрон ошибочно удаляется.
        ФИКС: current_best берётся из свежего evaluate() → current_best≈10
              → best_tmp(10) не > 10 → удаления нет → добавляется нейрон.
        """
        pop = ESPPopulation(input_size=4, hidden_size=3, output_size=1,
                            subpop_size=4, trials_per_individual=1)
        old_hidden = pop.hidden_size
        pop.burst_mutation()  # обнуляет cum_fitness / count_trials

        env = make_env(obs_size=4, reward=10.0)
        pop.adapt_structure(env, n_episodes=1)

        assert pop.hidden_size >= old_hidden, (
            f"Нейрон был ложно удалён: hidden {old_hidden} → {pop.hidden_size}. "
            f"current_best, вероятно, был 0.0 (из обнулённых cum_fitness после burst)."
        )

    def test_evaluate_called_for_baseline(self):
        """
        adapt_structure обязана вызвать evaluate() для получения current_best.
        Без этого baseline всегда 0, что ломает логику сравнения.
        """
        pop = ESPPopulation(input_size=4, hidden_size=3, output_size=1,
                            subpop_size=4, trials_per_individual=1)
        pop.burst_mutation()

        eval_calls = []
        original_evaluate = pop.evaluate

        def tracking_evaluate(e, n_episodes=1):
            eval_calls.append('called')
            return original_evaluate(e, n_episodes=n_episodes)

        pop.evaluate = tracking_evaluate
        pop.adapt_structure(make_env(obs_size=4, reward=5.0), n_episodes=1)

        assert len(eval_calls) >= 1, (
            "adapt_structure не вызвала evaluate() — current_best не может "
            "быть корректным."
        )


# ─────────────────────────────────────────────────────────────────────────────
# БАГ 2: при добавлении нейрона w_ho старых особей не должен расти
# ─────────────────────────────────────────────────────────────────────────────

class TestBug2GenomeExpansion:

    def test_genome_length_after_add_neuron(self):
        """
        Ожидаемая длина генома = input_size + new_hidden_size + output_size.

        БАГ: код делает new_ho = concat([ho, randn(output_size)]) → длина
             растёт до input + (h+1) + 2*output.
        ФИКС: new_vec = concat([ih, new_hh, ho]) — ho не трогаем.
        """
        in_s, h, out_s = 3, 2, 2
        pop = ESPPopulation(input_size=in_s, hidden_size=h, output_size=out_s,
                            subpop_size=4, trials_per_individual=1)
        env = make_env(obs_size=in_s, reward=-9999.0)
        force_add_branch(pop, env)
        pop.adapt_structure(env, n_episodes=1)

        if pop.hidden_size == h + 1:
            new_h = pop.hidden_size
            expected = in_s + new_h + out_s
            for sp_idx in range(h):  # старые подпопуляции
                for genome in pop.subpopulations[sp_idx]:
                    assert len(genome) == expected, (
                        f"Геном подпопуляции {sp_idx}: длина {len(genome)}, "
                        f"ожидалась {expected}. "
                        f"Вероятно, w_ho ошибочно расширен "
                        f"(+{out_s} лишних значений)."
                    )
        else:
            pytest.skip("Добавление нейрона не сработало — проверьте force_add_branch")

    def test_who_value_unchanged_after_add(self):
        """
        Значения w_ho старых особей не должны измениться при добавлении нейрона.

        БАГ: к w_ho дописываются случайные значения → w_ho растёт с output_size
             до 2*output_size.
        ФИКС: w_ho остаётся прежним.
        """
        in_s, h, out_s = 3, 2, 2
        pop = ESPPopulation(input_size=in_s, hidden_size=h, output_size=out_s,
                            subpop_size=4, trials_per_individual=1)

        # Запоминаем w_ho каждого генома до изменения
        whos_before = {}
        for sp_idx in range(h):
            for j, genome in enumerate(pop.subpopulations[sp_idx]):
                whos_before[(sp_idx, j)] = genome[in_s + h:].copy()

        env = make_env(obs_size=in_s, reward=-9999.0)
        force_add_branch(pop, env)
        pop.adapt_structure(env, n_episodes=1)

        if pop.hidden_size == h + 1:
            new_h = pop.hidden_size
            for sp_idx in range(h):
                for j, genome in enumerate(pop.subpopulations[sp_idx]):
                    ho_after = genome[in_s + new_h: in_s + new_h + out_s]
                    assert len(ho_after) == out_s, (
                        f"w_ho подпопуляции {sp_idx} особь {j}: "
                        f"длина {len(ho_after)}, ожидалась {out_s}."
                    )
                    assert np.allclose(ho_after, whos_before[(sp_idx, j)]), (
                        f"w_ho изменился при добавлении нейрона. "
                        f"До: {whos_before[(sp_idx, j)]}, после: {ho_after}."
                    )
        else:
            pytest.skip("Добавление нейрона не сработало")


# ─────────────────────────────────────────────────────────────────────────────
# БАГ 3: при удалении нейрона i из w_hh удаляется именно i-й элемент
# ─────────────────────────────────────────────────────────────────────────────

class TestBug3GenomeTrimming:

    def _setup_distinguishable_wh(self, pop):
        """Задаёт различимые значения w_hh: genome[input+k] = sp*1000 + k*10 + j."""
        in_s, h = pop.input_size, pop.hidden_size
        for sp_idx in range(h):
            for j in range(pop.subpop_size):
                for k in range(h):
                    pop.subpopulations[sp_idx][j][in_s + k] = float(
                        sp_idx * 1000 + k * 10 + j
                    )

    def _patch_eval_remove_neuron0(self):
        """Патч evaluate: 1-й вызов (baseline) → 1.0, остальные → 50.0.
        Это заставляет adapt_structure удалить нейрон 0 (первый улучшающий)."""
        call_count = [0]

        def patched(self_inner, env, n_episodes=1):
            call_count[0] += 1
            val = 1.0 if call_count[0] == 1 else 50.0
            return [np.full(self_inner.subpop_size, val)
                    for _ in range(self_inner.hidden_size)]

        return patched

    def test_correct_whh_element_removed(self):
        """
        При удалении нейрона i=0 из w_hh оставшихся геномов должен исчезнуть
        элемент с индексом 0, а не последний.

        БАГ: срез [:new_hidden] всегда отбрасывает последний элемент w_hh
             независимо от того, какой нейрон i удалён.
        ФИКС: геном обрезается через np.delete(hh, remove_idx).
        """
        in_s, h, out_s = 2, 4, 1
        pop = ESPPopulation(input_size=in_s, hidden_size=h, output_size=out_s,
                            subpop_size=3, trials_per_individual=1)
        self._setup_distinguishable_wh(pop)

        with patch.object(ESPPopulation, 'evaluate',
                          self._patch_eval_remove_neuron0()):
            pop.adapt_structure(make_env(in_s, out_s), n_episodes=1)

        assert pop.hidden_size == h - 1, "Нейрон 0 должен был быть удалён"

        new_h = pop.hidden_size
        for sp_idx in range(new_h):
            for j, genome in enumerate(pop.subpopulations[sp_idx]):
                hh_part = genome[in_s: in_s + new_h]
                assert len(hh_part) == new_h, \
                    f"Длина w_hh = {len(hh_part)}, ожидалось {new_h}"
                # Вес нейрона 0 для этой особи = sp_idx*1000 + 0*10 + j
                neuron0_weight = float(sp_idx * 1000 + j)
                assert neuron0_weight not in hh_part, (
                    f"w_hh подпопуляции {sp_idx} особь {j} содержит вес "
                    f"удалённого нейрона 0 ({neuron0_weight}): {hh_part}. "
                    f"Удалён последний элемент вместо i=0-го."
                )

    def test_who_not_corrupted_after_removal(self):
        """
        После удаления нейрона i срез w_ho не должен смещаться.

        БАГ: assemble_network использует [input + new_h : input + new_h + out]
             но геном не обрезан → срез начинается на 1 позицию раньше,
             захватывая последний элемент w_hh вместо первого w_ho.
        ФИКС: геном правильно обрезается → w_ho начинается ровно с позиции
              input + new_h.
        """
        in_s, h, out_s = 2, 3, 2
        pop = ESPPopulation(input_size=in_s, hidden_size=h, output_size=out_s,
                            subpop_size=3, trials_per_individual=1)

        # w_hh — положительные (1, 2, 3), w_ho — отрицательные (-1, -2)
        for sp_idx in range(h):
            for j in range(pop.subpop_size):
                genome = pop.subpopulations[sp_idx][j]
                for k in range(h):
                    genome[in_s + k] = float(k + 1)            # w_hh > 0
                for k in range(out_s):
                    genome[in_s + h + k] = float(-(k + 1))     # w_ho < 0

        with patch.object(ESPPopulation, 'evaluate',
                          self._patch_eval_remove_neuron0()):
            pop.adapt_structure(make_env(in_s, out_s), n_episodes=1)

        if pop.hidden_size == h - 1:
            new_h = pop.hidden_size
            for sp_idx in range(new_h):
                for genome in pop.subpopulations[sp_idx]:
                    ho_part = genome[in_s + new_h: in_s + new_h + out_s]
                    assert all(v < 0 for v in ho_part), (
                        f"w_ho = {ho_part} содержит неотрицательное значение. "
                        f"Вероятно, срез сдвинут: захвачен элемент из w_hh."
                    )
        else:
            pytest.skip("Удаление нейрона не сработало")


# ─────────────────────────────────────────────────────────────────────────────
# БАГ 4: в select_and_breed не должно быть гауссовой мутации (алг. 7.1)
# ─────────────────────────────────────────────────────────────────────────────

class TestBug4NoGaussianMutation:

    def test_upper_half_unchanged_with_cauchy_zero(self):
        """
        При alpha_cauchy=0:
          • Коши не изменяет никого (alpha=0)
          • Кроссинговер с одинаковыми геномами ничего не меняет
          • Верхняя половина должна остаться неизменной (Гаусс-цикл удалён).
        """
        np.random.seed(42)
        in_s, h, out_s = 2, 1, 1
        pop = ESPPopulation(input_size=in_s, hidden_size=h, output_size=out_s,
                            subpop_size=8, trials_per_individual=1,
                            alpha_cauchy=0.0)

        # Все геномы одинаковы → кроссинговер не меняет значений
        fixed = np.ones(in_s + h + out_s) * 5.0
        for j in range(8):
            pop.subpopulations[0][j] = fixed.copy()

        avg_fitness = [np.linspace(8, 1, 8)]  # убывающий fitness
        pop.select_and_breed(avg_fitness)

        # Верхняя половина (индексы 0–3 в sorted-порядке после select_and_breed)
        changed = sum(
            1 for genome in pop.subpopulations[0][:4]
            if not np.allclose(genome, 5.0)
        )
        assert changed == 0, (
            f"{changed}/4 геномов верхней половины изменены. "
            f"Гауссова мутация применяется ко всем особям (баг 4). "
            f"После фикса (удаление Гаусс-цикла) этого быть не должно."
        )

    def test_lower_half_mutated_by_cauchy(self):
        """
        Нижняя половина ДОЛЖНА изменяться Коши-мутацией (это корректно по алг. 7.1).
        Тест проверяет, что Коши-мутация сохраняется после фикса.
        """
        np.random.seed(0)
        in_s, h, out_s = 2, 1, 1
        pop = ESPPopulation(input_size=in_s, hidden_size=h, output_size=out_s,
                            subpop_size=8, trials_per_individual=1,
                            alpha_cauchy=10.0)     # большой сдвиг — точно изменится

        fixed = np.ones(in_s + h + out_s) * 5.0
        for j in range(8):
            pop.subpopulations[0][j] = fixed.copy()

        avg_fitness = [np.linspace(8, 1, 8)]
        pop.select_and_breed(avg_fitness)

        changed_lower = sum(
            1 for genome in pop.subpopulations[0][4:]
            if not np.allclose(genome, 5.0)
        )
        assert changed_lower > 0, (
            "Коши-мутация нижней половины не применилась. "
            "После фикса этот механизм должен остаться."
        )


# ─────────────────────────────────────────────────────────────────────────────
# БАГ 5: adapt_structure — однократный проход (не while-цикл)
# ─────────────────────────────────────────────────────────────────────────────

class TestBug5SinglePass:

    def test_at_most_one_neuron_removed_per_call(self):
        """
        Один вызов adapt_structure должен удалить не более 1 нейрона (алг. 7.3).

        БАГ: while-цикл → если каждое удаление «лучше», удаляются все нейроны
             до hidden_size=1.
        ФИКС: однократный for-проход → удаляется ровно 1 нейрон, затем выход.
        """
        pop = ESPPopulation(input_size=4, hidden_size=5, output_size=1,
                            subpop_size=4, trials_per_individual=1)
        old_hidden = pop.hidden_size  # 5
        call_count = [0]

        def always_better(self_inner, env, n_episodes=1):
            call_count[0] += 1
            # 1-й вызов: baseline (current_best = 1)
            # остальные: removal всегда лучше (50 > 1)
            val = 1.0 if call_count[0] == 1 else 50.0
            return [np.full(self_inner.subpop_size, val)
                    for _ in range(self_inner.hidden_size)]

        with patch.object(ESPPopulation, 'evaluate', always_better):
            pop.adapt_structure(make_env(4, 1), n_episodes=1)

        removed = old_hidden - pop.hidden_size
        assert removed <= 1, (
            f"Удалено {removed} нейронов за один вызов adapt_structure. "
            f"Ожидалось ≤ 1. Причина: while-цикл (баг 5)."
        )

    def test_adds_neuron_when_no_removal_helps(self):
        """
        Если ни одно удаление не улучшает — должен добавиться ровно 1 нейрон.
        """
        pop = ESPPopulation(input_size=4, hidden_size=3, output_size=1,
                            subpop_size=4, trials_per_individual=1)
        old_hidden = pop.hidden_size
        call_count = [0]

        def removal_worse(self_inner, env, n_episodes=1):
            call_count[0] += 1
            val = 100.0 if call_count[0] == 1 else 1.0  # baseline лучше
            return [np.full(self_inner.subpop_size, val)
                    for _ in range(self_inner.hidden_size)]

        with patch.object(ESPPopulation, 'evaluate', removal_worse):
            pop.adapt_structure(make_env(4, 1), n_episodes=1)

        assert pop.hidden_size == old_hidden + 1, (
            f"Ожидалось добавление 1 нейрона: {old_hidden} → {old_hidden + 1}, "
            f"получено {pop.hidden_size}."
        )
