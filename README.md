## Реализация алгоритма нейроэволюции ESP (Ensemble of Subpopulations) для задачи управления агентом (лунным кораблем). Рекуррентная сеть — 1 скрытый слой.

## Запуск

Обучение: 
```bash
python esp_lander.py --train --epochs 500 --hidden_size 12 --subpop_size 20
```

Тест: 
```bash
python esp_lander.py --test --load_weights model.pkl
```

Визуализация структуры:
```bash
python esp_lander.py --visualize_structure --load_weights model.pkl --outfile net.png
```

## Метрика

В качестве целевой метрики используется **среднее суммарное вознаграждение за эпизод**.


