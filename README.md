# Readme

## Краткое содержание

В финальный код вошли три модели - случайный лес из sklearn, бустинг над деревьями catboost и нейронная сеть на pytorch. Первая слаба (F1 = 0.941), вторая побила бенчмарки (F1 = 0.963), третья показала себя наилучшим образом (F1 = 0.967).

Обучал локально на GPU Nvidia GTX1070. Для воспроизведения нужны и python-библиотеки, и CUDA Toolkit. Обучение моделей завернуто в скрипты, в готовом виде они лежат в директории ./models/.

См. ноутбуки **explore_data.ipynb** и **look_at_models.ipynb**. Если интересны подробности, можно посмотреть и скрипты.

## Данные и задача

Данные - текст, представленный в формате BERT-эмбеддингов и меток класса. Исходный текст недоступен из-за конфиденциальности данных. Размерность: 90847*770. Количество классов: 14

Данные очень сильно не сбалансированы: самый крупный класс 7 содержит 61078 точек, самый мелкий класс 2 - 90.

Раз признаки построены state-of-the-art методом и количество данных большое, сразу нацелимся на обучение сложной модели. Отбор признаков, если он и необходим, оставим за алгоритмами. Будем считать, что это качественные и предобработанные данные.

В качестве тестовой выборки возьмем 25% данных. Кросс-валидация проводится так же. Для обучения нейросети размер теста будет 5%, потому что не будем дообучать ее на всех данных.

На Kaggle указано, что метрика - **mean F1**, но это не так. Результат валидации не близок к **f1_score(y_test, y_hat, average='macro')** из sklearn. Целевая метрика - **weighted F1**, где веса пропорциональны размерам классов.

В такой постановке самым важным из всех является класс с самой большой мощностью. При пометке всего набора данных классом 7 получаем **weighted F1 = 0.845** и **mean F1 (=macro F1) = 0.067**.

## Выбор классификаторов

Я буду использовать три модели, строящие сложные решающие правила и подходящие для многоклассовой классификации.

* RandomForestClassifier, sklearn
* CatBoostClassifier, catboost
* PyTorch: [nn.BatchNorm1d(770, *eps*=1e-05, *momentum*=0.1, *affine*=True),    nn.Linear(*in_features*=770, *out_features*=1024, *bias*=True),    nn.ReLU(),    nn.Dropout(*p*=0.5),    nn.Linear(*in_features*=1024, out_features*=256, *bias*=True),    nn.ReLU(),    nn.Linear(*in_features*=256, *out_features*=14, *bias*=True),     nn.LogSoftmax()]


### Случайный лес

Распараллелен, приспособлен для задач с высокой размерностью, пригоден для мультиклассификации, не требует шкалирования признаков.

#### Параметры

* Количество деревьев *n_estimators*

  Отчасти защищает от переобучения за счет того, что количество классификаторов растет. Не влияет на результат после высокого порога. Поэтому фиксируем на 300.

* Веса классов
  Оставим сбалансированными в виду оптимизируемой метрики: большие классы более важны.

* Количество признаков в узле дерева *max_features*

  Влияет на некоррелированность деревьев леса между собой. Для классификации по умолчанию берется *sqrt(features)*.

* Максимальная глубина дерева *max_depth*

  Слишком большая глубина приводит к переобучению леса. Экспериментально я подобрал порог ~20, где переобучение прекращается. Проверим.

* Максимальное количество листьев в дереве *max_leaf_nodes*

  Из всех оставляет листья с максимальным относительным приростом gini. Более гибкий вариант max_depth.

* Минимум количество точек, необходимый для разбиения листа *min_samples_split*

  По смыслу похоже на *max_depth*. Взял за дефолт значение 3.

Много экспериментировал руками, чтобы не перебирать огромную сетку параметров. Возможно, при подробном переборе можно было найти куда лучший набор параметров.

### CatBoost
Быстро развивающийся вариант бустинга над деревьями. Можно запускать на GPU. Показывает отличные результаты на практике.

Функция потерь: MultiClass

Метрика: TotalF1 (то же самое, что weighted F1)

Для предотвращения переобучения берем тестовую выборку как eval_set. Среди всех итераций обучения возьмем модель с лучшим weighted F1 на тесте.

#### Параметры

* Максимальное количество предикторов *iterations*

  Чем больше индекс предиктора, тем меньше он корректирует решающее правило.

* Скорость обучения *learning_rate*
  Определяет величину шага по градиенту целевой функции.

* Сила регуляризации *l2_leaf_reg*

  Коэффициент перед регуляризатором в целевой функции. Уменьшает переобучение.

* Максимальная глубина дерева *depth*

  Влияет на скорость обучения, сложность и качество модели. Лучшее значение~ 5-6. Чем больше - тем сложнее предикторы.

* Интенсивность бэггинга *bagging_temperature*

  Регулирует уровень случайности при бэггинге. Чем больше - тем больше случайности при назначении объектам весов. От 0 до +infty. При 1 веса объектов сэмплируются из экспоненциального распределения.

### Fully-connected Neural Network
Решил попробовать после успешного применения MLP. Несколько полносвязных слоев, ReLU функция в качестве активации. Для регуляризации используем Dropout, BatchNormalization для нормирования батчей данных. 

Функция потерь: на последнем слое использован logsoftmax, конечное решение выбирается простым максимумом по выходу последнего слоя (logsoftmax - аналог логарифма от вероятности). Функция потерь классификатора - negative log likelihood loss (NLLLoss). Т.е. оптимизируется правдоподобие.

Метрика: weighted F1

#### Параметры

* Количество слоев и их размер.
  Чем больше нейронов в сети, тем более сложные функции она может приближать, но требует больше данных и может переобучиться.

* Функция активации.
  Влияет на класс функций, которые может аппроксимировать сеть, на качество обучения и т.д. Был выбран ReLU.

* Нормализация данных.
  Был добавлен слой BatchNormalization для увеличения стабильности и производительности (нормирует данные).

* Скорость обучения *learning_rate*
  Определяет величину шага по градиенту целевой функции. Скорость обучения (learning rate) изменялась методом CosineAnnealingLR. Вместе с числом эпох обучения *n_epochs* задает суммарное количество сэмплов, на которых обучается модель.

* Регуляризация
  Слои Dropout были выбраны для регуляризации и предотвращения перебучения. 

* Оптимизатор
  Влияет на качество и скорость обучения. В качестве оптимизатора были использованы Adam и RMSProp (в разных экспериментах).

## Как обучить и запустить

### Python dependencies

Последняя версия anaconda с python 3.7, туда входят sklearn, pandas, numpy, matplotlib, etc. Кроме этого: pytorch, catboost, ipypb, pandas-ml

### GPU-related dependencies

Для вычисления на GPU нужны драйверы Nvidia + CUDA Toolkit. Установка pytorch также зависит от системы, оборудования и версии CUDA. Я использовал CUDA 10.0.13 под Windows.

### Воспроизведение обучения моделей

Обучение моделей с оптимальными параметрами производится запуском скрипта train_models. Обучение Catboost с указанными параметрами на Nvidia GTX 1070 занимает ~2 минуты, нейросети - столько же.

### Загрузка уже обученных моделей

Лучшие модели случайного леса, catboost и нейросети лежат в *./models/* и загружаются стандартными функциями соотв. библиотек. Самой точной из моделей является нейросеть c weighted F1 = 0.96772 с самым высоким F1 на тесте: 0.965. 

После загрузки при помощи 
```python
nn = torch.load('./models/nn.pth') 
```
можно предсказывать на новых тестовых данных.