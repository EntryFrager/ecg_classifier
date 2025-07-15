# Условия эксперимента

## Гиперпараметры обучения на ЭКГ-сигналах

Общие для всех классов:

|**HyperParameters**|**Value**|
|-------------|--------|
|**sampling rate при записи ЭКГ-сигналов**| 100Гц|
|**Архитектура нейросети**| ResNet1d|
|**epochs**|30|
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

| **Гиперпараметр**        | **sinus**     | **arrit** | **tach**  | **brad**  | **afib**  |
|----------------------|-----------|-------|-------|-------|-------|
|**Batch size**            | 32|64|128|32|128|
|**learning rate**         | 0.001|0.001|0.0001|0.0001|0.01|

## Гиперпараметры обучения на ЭКГ-сигналах с метаданными

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