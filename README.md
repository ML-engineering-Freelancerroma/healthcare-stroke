# Исследование влияния SMOTENC и RandomUnderSampler на задачу бинарной классификации при сильном дисбалансе классов

Этот репозиторий содержит небольшое исследование влияния методов **SMOTENC** и **RandomUnderSampler** (библиотека `imblearn`) на данные с выраженным дисбалансом целевого класса в задаче бинарной классификации.

Датасет: [Healthcare Dataset Stroke Data](https://www.kaggle.com/datasets/aouatifcherdid/healthcare-dataset-stroke-data)

## Основные этапы работы

- Данные разделены на валидационную и тестовую выборки в соотношении **75:25** с сохранением пропорции целевого класса  
  → файл: `healthcare-stroke-prepare-for-ML.ipynb`

- Выполнена обработка признаков различными способами:  
  - one-hot encoding  
  - label encoding  
  - ручное кодирование  
  → файл: `feat_prepare.ipynb`

## Важные метрики

Для задач такого типа приоритетными являются:

- **Recall**  
- **ROC-AUC**  
- **PR-AUC**  

(важно выявить как можно больше истинно положительных случаев — пациентов с риском инсульта — при контроле ложных срабатываний)

## Обучение моделей

### 1. На исходных (несбалансированных) данных
Перебор гиперпараметров с помощью `GridSearch` + `StratifiedKFold`:

- `KNeighborsClassifier`
- `DecisionTreeClassifier`
- `RandomForestClassifier` (scikit-learn)
- `CatBoostClassifier` (catboost)
- `XGBClassifier` (xgboost)
- `MLPClassifier` (sklearn.neural_network)

→ файл: `train_orig.ipynb`

### 2. С применением SMOTENC
→ файл: `train_SMOTENC.ipynb`

### 3. С применением RandomUnderSampler
→ файл: `train_RUS.ipynb`

## Выводы по результатам

По большинству метрик **RandomUnderSampler** показал себя лучше при стандартных/типовых настройках моделей.

Более тонкая настройка гиперпараметров может дать заметно лучший результат, но требует значительных вычислительных ресурсов и времени.

## Финальная итерация

Повторное обучение наиболее перспективных моделей с расширенной сеткой гиперпараметров через `RandomizedSearchCV`:

- `RandomForestClassifier`
- `XGBClassifier`
- `CatBoostClassifier`

Далее проведено ансамблирование этих моделей с помощью `VotingClassifier` → получен небольшой прирост по метрике **PR-AUC**.

## Итог

Проведено тестирование на отложенной тестовой выборке (данные, не участвовавшие в обучении и подборе гиперпараметров).