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
|1649|89|278|160|

|**metrics**|**value**|
|**sensitivity**|0.911553|
|**specificity**|0.757493|
|**precision**|0.948792|
|**f1_score**|0.929800|

**ROC AUC:** 0.9226

**Classification report from sklearn:**
| **Class**        | **Precision** | **Recall** | **F1-score** | **Support** |
|--------------|-----------|--------|----------|---------|
| **0.0**      | 0.63      | 0.76   | 0.69     | 367     |
| **1.0**      | 0.95      | 0.91   | 0.93     | 1809    |
|              |           |        |          |         |
| **Accuracy** |           |        | 0.89     | 2176    |
| **Macro avg**| 0.79      | 0.83   | 0.81     | 2176    |
| **Weighted avg** | 0.90  | 0.89   | 0.89     | 2176    |

**Test Loss:** 0.1200

---

### Аритмия:

**Confusion matrix:**
|**TP**|**FP**|**TN**|**FN**|
|76|1605|483|12|

|**metrics**|**value**|
|**sensitivity**|0.863636|
|**specificity**|0.231322|
|**precision**|0.045211|
|**f1_score**|0.085924|

**ROC AUC:** 0.5933

**Classification report from sklearn:**
| **Class**        | **Precision** | **Recall** | **F1-score** | **Support** |
|--------------|-----------|--------|----------|---------|
| **0.0**      | 0.98      | 0.23   | 0.37     | 2088    |
| **1.0**      | 0.05      | 0.86   | 0.09     | 88      |
|              |           |        |          |         |
| **Accuracy** |           |        | 0.26     | 2176    |
| **Macro avg**| 0.51      | 0.55   | 0.23     | 2176    |
| **Weighted avg** | 0.94  | 0.26   | 0.36     | 2176    |

**Test Loss:** 1.2805

---

### Тахикардия:

**Confusion matrix:**
|**TP**|**FP**|**TN**|**FN**|
|81|118|1975|2|

|**metrics**|**value**|
|**sensitivity**|0.975904|
|**specificity**|0.943622|
|**precision**|0.407035|
|**f1_score**|0.574468|

**ROC AUC:** 0.9879

**Classification report from sklearn:**
| **Class**        | **Precision** | **Recall** | **F1-score** | **Support** |
|--------------|-----------|--------|----------|---------|
| **0.0**      | 1.00      | 0.94   | 0.97     | 2093    |
| **1.0**      | 0.41      | 0.98   | 0.57     | 83      |
|              |           |        |          |         |
| **Accuracy** |           |        | 0.94     | 2176    |
| **Macro avg**| 0.70      | 0.96   | 0.77     | 2176    |
| **Weighted avg** | 0.98  | 0.94   | 0.96     | 2176    |

**Test Loss:** 0.2461

---

### Брадикардия:

**Confusion matrix:**
|**TP**|**FP**|**TN**|**FN**|
|61|344|1768|3|

|**metrics**|**value**|
|**sensitivity**|0.953125|
|**specificity**|0.837121|
|**precision**|0.150617|
|**f1_score**|0.260128|

**ROC AUC:** 0.9513

**Classification report from sklearn:**
| **Class**        | **Precision** | **Recall** | **F1-score** | **Support** |
|--------------|-----------|--------|----------|---------|
| **0.0**      | 1.00      | 0.84   | 0.91     | 2112    |
| **1.0**      | 0.15      | 0.95   | 0.26     | 64      |
|              |           |        |          |         |
| **Accuracy** |           |        | 0.84     | 2176    |
| **Macro avg**| 0.57      | 0.90   | 0.59     | 2176    |
| **Weighted avg** | 0.97  | 0.84   | 0.89     | 2176    |

**Test Loss:** 0.5556

---

### Фибрилляция предсердий:

**Confusion matrix:**
|**TP**|**FP**|**TN**|**FN**|
|137|57|1968|14|

|**metrics**|**value**|
|**sensitivity**|0.907285|
|**specificity**|0.971852|
|**precision**|0.706186|
|**f1_score**|0.794203|

**ROC AUC:** 0.9767

**Classification report from sklearn:**
| **Class**        | **Precision** | **Recall** | **F1-score** | **Support** |
|--------------|-----------|--------|----------|---------|
| **0.0**      | 0.99      | 0.97   | 0.98     | 2025    |
| **1.0**      | 0.71      | 0.91   | 0.79     | 151     |
|              |           |        |          |         |
| **Accuracy** |           |        | 0.97     | 2176    |
| **Macro avg**| 0.85      | 0.94   | 0.89     | 2176    |
| **Weighted avg** | 0.97  | 0.97   | 0.97     | 2176    |

**Test Loss:** 0.3583


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