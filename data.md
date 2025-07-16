# Условия эксперимента

## Общие гиперпараметры

Общие для всех классов:

|**HyperParameters**|**Value**|
|-------------|--------|
|**sampling rate при записи ЭКГ-сигналов**| 100Гц|
|**Архитектура нейросети**| ResNet1d|
|**epochs**|30|
|**Batch size**|64|
|**Optimizer**|Adam|
|**Optimizer weight decay**| 1e-6|
|**Loss function**|BCEWithLogitsLoss|
|**Scheduler**|ReduceLROnPlateau|
|**Scheduler mode**|min|
|**Scheduler factor**|0.6|
|**Scheduler patience**|3|
|**EarlyStopping patience**|8|
|**dropout‑rate в head**|0.25|
|**dropout‑rate в backbone**|0.1|
|**dropout-rate в metadata (при добавлении метаданных)**|0.1|

В связи с тем, что синусовый ритм является обычным состояним человека, то его бинарная классификация **1** фактически означает, что он здоров, а **0** - не здоров. То есть необходимо смотреть при обучении на **specificity**, чтобы получить низкий **FP**. Для этого коэффициенты **alpha** и **beta** отвечающие соответственно за **sensitivity** и **specificity** равны **0.2**, **0.8**. А для остальных классов наоборот, $alpha = 0.8$ и $beta = 0.2$

## Подобранные гиперпараметры для всех классов при обучении на ЭКГ-сигналах

| **Гиперпараметр**        | **sinus**     | **arrit** | **tach**  | **brad**  | **afib**  |
|----------------------|-----------|-------|-------|-------|-------|
|**learning rate**         | 0.001|0.001|0.0001|0.0001|0.01|

## Подобранные гиперпараметры для всех классов при обучении на ЭКГ-сигналах с метаданными

| **Гиперпараметр**        | **sinus**     | **arrit** | **tach**  | **brad**  | **afib**  |
|----------------------|-----------|-------|-------|-------|-------|
|**learning rate**         | 0.001|0.001|0.0001|0.0001|0.01|

# Результаты

## Обучение на ЭКГ-сигналах

### Синусовый ритм:

**Confusion matrix:**
|**TP**|**FP**|**TN**|**FN**|
|1769|137|240|57|

|**metrics**|**value**|
|**sensitivity**|0.97|
|**specificity**|0.64|
|**precision**|0,93|
|**f1_score**|0.95|

**ROC AUC:** 0.92

**Classification report from sklearn:**
| **Class**        | **Precision** | **Recall** | **F1-score** | **Support** |
|--------------|-----------|--------|----------|---------|
| **0.0**      | 0.81      | 0.64   | 0.71     | 377     |
| **1.0**      | 0.93      | 0.97   | 0.95     | 1826    |
|              |           |        |          |         |
| **Accuracy** |           |        | 0.91     | 2203    |
| **Macro avg**| 0.87      | 0.80   | 0.83     | 2203    |
| **Weighted avg** | 0.91  | 0.91   | 0.91     | 2203    |

**Test Loss:** 0.1215

---

### Аритмия:

**Confusion matrix:**
|**TP**|**FP**|**TN**|**FN**|
|32|353|1759|59|

|**metrics**|**value**|
|**sensitivity**|0.35|
|**specificity**|0.83|
|**precision**|0.08|
|**f1_score**|0.13|

**ROC AUC:** 0.64

**Classification report from sklearn:**
| **Class**        | **Precision** | **Recall** | **F1-score** | **Support** |
|--------------|-----------|--------|----------|---------|
| **0.0**      | 0.97      | 0.83   | 0.90     | 2112    |
| **1.0**      | 0.08      | 0.35   | 0.13     | 91      |
|              |           |        |          |         |
| **Accuracy** |           |        | 0.81     | 2203    |
| **Macro avg**| 0.53      | 0.59   | 0.51     | 2203    |
| **Weighted avg** | 0.93  | 0.81   | 0.86     | 2203    |

**Test Loss:** 1.3088

---

### Тахикардия:

**Confusion matrix:**
|**TP**|**FP**|**TN**|**FN**|
|77|19|2100|7|

|**metrics**|**value**|
|**sensitivity**|0.92|
|**specificity**|0.99|
|**precision**|0.80|
|**f1_score**|0.86|

**ROC AUC:** 0.99

**Classification report from sklearn:**
| **Class**        | **Precision** | **Recall** | **F1-score** | **Support** |
|--------------|-----------|--------|----------|---------|
| **0.0**      | 1.00      | 0.99   | 0.99     | 2119    |
| **1.0**      | 0.80      | 0.92   | 0.86     | 84      |
|              |           |        |          |         |
| **Accuracy** |           |        | 0.99     | 2203    |
| **Macro avg**| 0.90      | 0.95   | 0.92     | 2203    |
| **Weighted avg** | 0.99  | 0.99   | 0.99     | 2203    |

**Test Loss:** 0.2406

---

### Брадикардия:

**Confusion matrix:**
|**TP**|**FP**|**TN**|**FN**|
|35|17|2122|29|

|**metrics**|**value**|
|**sensitivity**|0.55|
|**specificity**|0.99|
|**precision**|0.67|
|**f1_score**|0.60|

**ROC AUC:** 0.95

**Classification report from sklearn:**
| **Class**        | **Precision** | **Recall** | **F1-score** | **Support** |
|--------------|-----------|--------|----------|---------|
| **0.0**      | 0.99      | 0.99   | 0.99     | 2139    |
| **1.0**      | 0.67      | 0.55   | 0.60     | 64      |
|              |           |        |          |         |
| **Accuracy** |           |        | 0.98     | 2203    |
| **Macro avg**| 0.83      | 0.77   | 0.80     | 2203    |
| **Weighted avg** | 0.98  | 0.98   | 0.98     | 2203    |

**Test Loss:** 0.5967

---

### Фибрилляция предсердий:

**Confusion matrix:**
|**TP**|**FP**|**TN**|**FN**|
|133|28|2108|24|

|**metrics**|**value**|
|**sensitivity**|0.85|
|**specificity**|0.99|
|**precision**|0.83|
|**f1_score**|0.84|

**ROC AUC:** 0.98

**Classification report from sklearn:**
| **Class**        | **Precision** | **Recall** | **F1-score** | **Support** |
|--------------|-----------|--------|----------|---------|
| **0.0**      | 0.99      | 0.99   | 0.99     | 2046    |
| **1.0**      | 0.83      | 0.85   | 0.84     | 157     |
|              |           |        |          |         |
| **Accuracy** |           |        | 0.98     | 2203    |
| **Macro avg**| 0.91      | 0.92   | 0.91     | 2203    |
| **Weighted avg** | 0.98  | 0.98   | 0.98     | 2203    |

**Test Loss:** 0.3697

## Обучение на ЭКГ-сигналах с метаданными

**Confusion matrix:**
|**TP**|**FP**|**TN**|**FN**|
|1596|64|303|213|

|**metrics**|**value**|
|**sensitivity**|0.882255|
|**specificity**|0.825613|
|**precision**|0.961446|
|**f1_score**|0.920150|

**ROC AUC:** 0.9176

**Classification report from sklearn:**
| **Class**        | **Precision** | **Recall** | **F1-score** | **Support** |
|--------------|-----------|--------|----------|---------|
| **0.0**      | 0.59      | 0.83   | 0.69     | 367     |
| **1.0**      | 0.96      | 0.88   | 0.92     | 1809    |
|              |           |        |          |         |
| **Accuracy** |           |        | 0.87     | 2176    |
| **Macro avg**| 0.77      | 0.85   | 0.80     | 2176    |
| **Weighted avg** | 0.90  | 0.87   | 0.88     | 2176    |

**Test Loss:** 0.1468

---

### Аритмия:

**Confusion matrix:**
|**TP**|**FP**|**TN**|**FN**|
|52|427|1661|36|

|**metrics**|**value**|
|**sensitivity**|0.590909|
|**specificity**|0.795498|
|**precision**|0.108559|
|**f1_score**|0.183422|

**ROC AUC:** 0.7202

**Classification report from sklearn:**
| **Class**        | **Precision** | **Recall** | **F1-score** | **Support** |
|--------------|-----------|--------|----------|---------|
| **0.0**      | 0.98      | 0.80   | 0.88     | 2088    |
| **1.0**      | 0.11      | 0.59   | 0.18     | 88      |
|              |           |        |          |         |
| **Accuracy** |           |        | 0.79     | 2176    |
| **Macro avg**| 0.54      | 0.69   | 0.53     | 2176    |
| **Weighted avg** | 0.94  | 0.79   | 0.85     | 2176    |

**Test Loss:** 1.4541


---

### Тахикардия:

**Confusion matrix:**
|**TP**|**FP**|**TN**|**FN**|
|81|80|2013|2|

|**metrics**|**value**|
|**sensitivity**|0.975904|
|**specificity**|0.961777|
|**precision**|0.503106|
|**f1_score**|0.663934|

**ROC AUC:** 0.9909

**Classification report from sklearn:**
| **Class**        | **Precision** | **Recall** | **F1-score** | **Support** |
|--------------|-----------|--------|----------|---------|
| **0.0**      | 1.00      | 0.96   | 0.98     | 2093    |
| **1.0**      | 0.50      | 0.98   | 0.66     | 83      |
|              |           |        |          |         |
| **Accuracy** |           |        | 0.96     | 2176    |
| **Macro avg**| 0.75      | 0.97   | 0.82     | 2176    |
| **Weighted avg** | 0.98  | 0.96   | 0.97     | 2176    |

**Test Loss:** 0.2229

---

### Брадикардия:

**Confusion matrix:**
|**TP**|**FP**|**TN**|**FN**|
|59|230|1882|5|

|**metrics**|**value**|
|**sensitivity**|0.921875|
|**specificity**|0.891098|
|**precision**|0.204152|
|**f1_score**|0.334278|

**ROC AUC:** 0.9664

**Classification report from sklearn:**
| **Class**        | **Precision** | **Recall** | **F1-score** | **Support** |
|--------------|-----------|--------|----------|---------|
| **0.0**      | 1.00      | 0.89   | 0.94     | 2112    |
| **1.0**      | 0.20      | 0.92   | 0.33     | 64      |
|              |           |        |          |         |
| **Accuracy** |           |        | 0.89     | 2176    |
| **Macro avg**| 0.60      | 0.91   | 0.64     | 2176    |
| **Weighted avg** | 0.97  | 0.89   | 0.92     | 2176    |

**Test Loss:** 0.4468

---

### Фибрилляция предсердий:

**Confusion matrix:**
|**TP**|**FP**|**TN**|**FN**|
|142|76|1949|9|

|**metrics**|**value**|
|**sensitivity**|0.940397|
|**specificity**|0.962469|
|**precision**|0.651376|
|**f1_score**|0.769648|

**ROC AUC:** 0.9704

**Classification report from sklearn:**
| **Class**        | **Precision** | **Recall** | **F1-score** | **Support** |
|--------------|-----------|--------|----------|---------|
| **0.0**      | 1.00      | 0.96   | 0.98     | 2025    |
| **1.0**      | 0.65      | 0.94   | 0.77     | 151     |
|              |           |        |          |         |
| **Accuracy** |           |        | 0.96     | 2176    |
| **Macro avg**| 0.82      | 0.95   | 0.87     | 2176    |
| **Weighted avg** | 0.97  | 0.96   | 0.96     | 2176    |

**Test Loss:** 0.4177