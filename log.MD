## 17/11/2023

10:13 running MNIST09 with:
- gamma = 2   
- class_n = 250 
- seed = 123 
- test_size=.4 
- hyperparameter tuning fold = 4 
- hyperparameter tuning n_experiments = 8
- hyperparameter tuning test_size = 0.6

total of 500 data 250 each class, train 300 test 200.

**For hyperparameter tuning:**

Each training set includes 120 samples 

Each validation set includes 180 samples

**hyperparameter tuning results**

| B  | K | xi | gamma | trainacc | trainauroc | trainauprc | testacc | testauroc | testauprc |
|----|---|----|-------|----------|------------|------------|---------|-----------|-----------|
| 2  | 2 | 1  | 2.000 | 0.8958   | 0.9781     | 0.9800     | 0.9306  | 0.9850    | 0.9868    |
| 2  | 2 | 2  | 2.000 | 0.9312   | 0.9832     | 0.9848     | 0.9236  | 0.9864    | 0.9876    |
| 2  | 3 | 1  | 2.000 | 0.9354   | 0.9839     | 0.9845     | 0.9375  | 0.9877    | 0.9886    |
| 2  | 3 | 2  | 2.000 | 0.9167   | 0.9780     | 0.9765     | 0.9347  | 0.9826    | 0.9843    |
| 10 | 2 | 1  | 2.000 | 0.8708   | 0.9330     | 0.9247     | 0.8931  | 0.9489    | 0.9522    |
| 10 | 2 | 2  | 2.000 | 0.8937   | 0.9731     | 0.9716     | 0.9194  | 0.9741    | 0.9739    |
| 10 | 3 | 1  | 2.000 | 0.8104   | 0.8774     | 0.8689     | 0.7931  | 0.8518    | 0.8364    |
| 10 | 3 | 2  | 2.000 | 0.8271   | 0.9036     | 0.8829     | 0.7861  | 0.8857    | 0.8582    |

best params are: B = 2 K = 3 xi = 1 gamma = 2.0

**For training**

- experiments = 4
- slice = 50      # Number of samples to use for the training
- fold = 6

**For testing**

- experiments=4
- slices=6  # Folds

**Results**
- **accuracy** = 0.925
- confusion matrix:

|   | **0**  |**1**  |
|---|----|-----|
| **0** | 45 | 4.5 |
| **1** | 3  | 47  |


## 19/11/2023


13.39 running MNIST38 with:
- gamma = 2   
- class_n = 250 
- seed = 123 
- test_size=.4 
- hyperparameter tuning fold = 4 
- hyperparameter tuning n_experiments = 8
- hyperparameter tuning test_size = 0.6

total of 500 data 250 each class, train 300 test 200.

**For hyperparameter tuning:**

Each training set includes 120 samples 

Each validation set includes 180 samples

**hyperparameter tuning results**

| B  | K | xi | gamma | trainacc | trainauroc | trainauprc | testacc | testauroc | testauprc |
|----|---|----|-------|----------|------------|------------|---------|-----------|-----------|
| 2  | 2 | 1  | 2.000 | 0.8333   | 0.8980     | 0.8650     | 0.8500  | 0.9150    | 0.8672    |
| 2  | 2 | 2  | 2.000 | 0.8500   | 0.8660     | 0.8183     | 0.8667  | 0.8990    | 0.8710    |
| 2  | 3 | 1  | 2.000 | 0.8667   | 0.9083     | 0.8391     | 0.8722  | 0.9372    | 0.9067    |
| 2  | 3 | 2  | 2.000 | 0.8521   | 0.8886     | 0.8521     | 0.8764  | 0.9103    | 0.8964    |
| 10 | 2 | 1  | 2.000 | 0.6854   | 0.7696     | 0.7340     | 0.6750  | 0.7591    | 0.7271    |
| 10 | 2 | 2  | 2.000 | 0.6729   | 0.7522     | 0.6990     | 0.6917  | 0.7770    | 0.6844    |
| 10 | 3 | 1  | 2.000 | 0.6562   | 0.7105     | 0.7340     | 0.6500  | 0.7102    | 0.7472    |
| 10 | 3 | 2  | 2.000 | 0.7625   | 0.8211     | 0.7732     | 0.7847  | 0.8520    | 0.8350    |

best params are: B = 2 K = 3 xi = 1 gamma = 2.0

**For training**

- experiments = 4
- slice = 50      # Number of samples to use for the training
- fold = 6

**For testing**

- experiments=4
- slices=6  # Folds

**Results**
- **accuracy** = 0.795
- confusion matrix:
 
|  | **0**  | **1** |
|---|----|-----|
| **0**| 28 | 1 |
|**1** | 20 | 51| 

## 20/11/2023


9:18 running MNIST47 with:
- gamma = 2   
- class_n = 250 
- seed = 123 
- test_size=.4 
- hyperparameter tuning fold = 4 
- hyperparameter tuning n_experiments = 8
- hyperparameter tuning test_size = 0.6

total of 500 data 250 each class, train 300 test 200.

**For hyperparameter tuning:**

Each training set includes 120 samples 

Each validation set includes 180 samples

**hyperparameter tuning results**

| B  | K | xi | gamma | trainacc | trainauroc | trainauprc | testacc | testauroc | testauprc |
|----|---|----|-------|----------|------------|------------|---------|-----------|-----------|
| 2  | 2 | 1  | 2.000 | 0.9146   | 0.9688     | 0.9727     | 0.8931  | 0.9547    | 0.9595    |
| 2  | 2 | 2  | 2.000 | 0.8792   | 0.9597     | 0.9646     | 0.8792  | 0.9612    | 0.9615    |
| 2  | 3 | 1  | 2.000 | 0.9354   | 0.9742     | 0.9795     | 0.9014  | 0.9674    | 0.9708    |
| 2  | 3 | 2  | 2.000 | 0.9250   | 0.9772     | 0.9790     | 0.9111  | 0.9684    | 0.9706    |
| 10 | 2 | 1  | 2.000 | 0.7833   | 0.9015     | 0.9030     | 0.8306  | 0.9227    | 0.9226    |
| 10 | 2 | 2  | 2.000 | 0.8542   | 0.9486     | 0.9526     | 0.8736  | 0.9428    | 0.9450    |
| 10 | 3 | 1  | 2.000 | 0.7500   | 0.8701     | 0.8732     | 0.7167  | 0.8249    | 0.8378    |
| 10 | 3 | 2  | 2.000 | 0.8292   | 0.9182     | 0.9264     | 0.7972  | 0.8994    | 0.9061    |

The best hyperparameters are B = 2 K = 3 xi = 1 gamma = 2.0

**For training**

- experiments = 4
- slice = 50      # Number of samples to use for the training
- fold = 6

**For testing**

- experiments=4
- slices=6  # Folds

**Results**
- **accuracy** = 0.96
- confusion matrix:
 
|  | **0**  | **1** |
|---|----|-----|
| **0**| 46 | 1 |
|**1** | 2 | 51| 

## 21/11/2023

16:30 running MNIST36 with:
- gamma = 2   
- class_n = 250 
- seed = 123 
- test_size=.4 
- hyperparameter tuning fold = 4 
- hyperparameter tuning n_experiments = 8
- hyperparameter tuning test_size = 0.6

total of 500 data 250 each class, train 300 test 200.

**For hyperparameter tuning:**

Each training set includes 120 samples 

Each validation set includes 180 samples

**hyperparameter tuning results**

| B  | K | xi | gamma | trainacc | trainauroc | trainauprc | testacc | testauroc | testauprc |
|----|---|----|-------|----------|------------|------------|---------|-----------|-----------|
| 2  | 2 | 1  | 2.000 | 0.8896   | 0.9613     | 0.9566     | 0.8903  | 0.9375    | 0.9264    |
| 2  | 2 | 2  | 2.000 | 0.8937   | 0.9675     | 0.9640     | 0.8972  | 0.9648    | 0.9602    |
| 2  | 3 | 1  | 2.000 | 0.9062   | 0.9630     | 0.9608     | 0.8944  | 0.9519    | 0.9437    |
| 2  | 3 | 2  | 2.000 | 0.9229   | 0.9679     | 0.9677     | 0.9028  | 0.9527    | 0.9429    |
| 10 | 2 | 1  | 2.000 | 0.8292   | 0.9021     | 0.9028     | 0.8194  | 0.8918    | 0.8975    |
| 10 | 2 | 2  | 2.000 | 0.8042   | 0.8930     | 0.9074     | 0.8028  | 0.8955    | 0.9043    |
| 10 | 3 | 1  | 2.000 | 0.8354   | 0.9321     | 0.9328     | 0.8250  | 0.9264    | 0.9338    |
| 10 | 3 | 2  | 2.000 | 0.8813   | 0.9366     | 0.9483     | 0.8722  | 0.9436    | 0.9457    |

The best hyperparameters are B = 2 K = 2 xi = 2 gamma = 2.0


**For training**

- experiments = 4
- slice = 50      # Number of samples to use for the training
- fold = 6

**For testing**

- experiments=4
- slices=6  # Folds

**Results**

- **accuracy** = 0.875
- confusion matrix:
 
|  | **0**  | **1** |
|---|----|-----|
| **0**| 37 | 1 |
|**1** | 11 | 51| 
