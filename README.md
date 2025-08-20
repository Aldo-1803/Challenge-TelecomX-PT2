# Telecom X – Parte 2: Predicción de Cancelación (Churn)

> **Rol:** Analista Junior de Machine Learning  
> **Objetivo:** Construir un pipeline de ML para predecir la cancelación de clientes (*churn*) y proponer acciones de negocio basadas en los factores más influyentes.

---

## 🧭 Resumen del proyecto

- **Problema:** Anticipar qué clientes tienen mayor probabilidad de cancelar su servicio en Telecom X.  
- **Output:** Modelos de clasificación + análisis interpretativo + recomendaciones de retención.  
- **Dataset:** Versión **limpia** y **estandarizada** de la Parte 1 → `datos_tratados.csv` (sin identificadores, columnas relevantes).

---

## ⚙️ Requisitos

- Python 3.12.11  
- Librerías principales:  
  `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `matplotlib`, `seaborn`

## 🚀 Cómo ejecutar

  1. Colocar data/datos_tratados.csv.
  2. Abrir el notebook notebooks/01_telecomx_parte2_modelado.ipynb.
  3. Ejecutar todas las celdas en orden.

El notebook realiza:

  * Carga de datos tratados.
  * Eliminación de columnas irrelevantes (customerid).
  * Separación de X (features) y y (target churn).
  * One-Hot Encoding a categóricas y StandardScaler a numéricas (via ColumnTransformer).
  * Verificación de balance de clases (≈ 73% no churn / 27% churn).
  * Opcional: SMOTE para balancear.
  * Train/Test split (70/30, stratify=y).
  * Entrenamiento de Regresión Logística y Random Forest.
  * Evaluación con Accuracy, Precision, Recall, F1, ROC-AUC y Matriz de Confusión.
  * Interpretación de coeficientes (LR) e importancia de variables (RF).
  * Gráficos comparativos de métricas e importancia de variables.

## 🧪 Modelos y evaluación

Se entrenaron dos enfoques complementarios:

  * Regresión Logística (sensible a escala, interpretable).
  * Random Forest (árboles, robusto a escala, no lineal).
    
| Modelo              | Accuracy | Precision (churn) | Recall (churn) | F1 (churn) | ROC-AUC |
| ------------------- | -------: | ----------------: | -------------: | ---------: | ------: |
| Regresión Logística |    0.798 |             0.640 |          0.547 |      0.590 |   0.840 |
| Random Forest       |    0.784 |             0.618 |          0.485 |      0.544 |   0.826 |

Lectura recomendada: en churn, priorizar Recall y F1-score de la clase positiva (clientes que cancelan), ya que el costo de no detectarlos es mayor que el de un falso positivo.

## 🔍 Factores que más influyen

### Regresión Logística – aumentan el riesgo (coef. +):
- `account_chargestotal` (gasto total)  
- `internet_internetservice_fiber optic`  
- `account_paymentmethod_electronic check`  
- `phone_multiplelines_yes`  
- `account_paperlessbilling`  

### Regresión Logística – reducen el riesgo (coef. –):
- `customer_tenure` (antigüedad)  
- `account_contract_two year`  
- `internet_internetservice_no`  
- `account_contract_one year`  
- `internet_techsupport_yes`  

### Random Forest – top importancias:
1. `account_chargestotal`  
2. `customer_tenure`  
3. `account_chargesmonthly`  
4. `cuentas_diarias`  
5. `internet_internetservice_fiber optic`  

📌 **Conclusión interpretativa:**  
El tipo de contrato, la antigüedad del cliente y los costos (mensuales/totales), junto con el tipo de servicio de internet y el método de pago, son los factores más determinantes para predecir la cancelación.

---

## ⚖️ Balance de clases
- Distribución original: ~**73% no churn / 27% churn**.  
- Se ofrece pipeline con **SMOTE** (opcional) para mejorar Recall en la clase minoritaria.  
- Alternativa: `class_weight` en modelos lineales o basados en árboles.

---

## 🧩 Consideraciones de overfitting/underfitting

- **Random Forest:** puede sobreajustarse si no se limitan parámetros como `max_depth`, `min_samples_leaf`, `n_estimators`, `max_features`.  
- **Regresión Logística:** puede subajustar (underfitting) si las relaciones son no lineales o falta ingeniería de variables. Ajustar regularización (`C`) o interacciones de features.

---

## 🗺️ Roadmap de mejora

- **Búsqueda de hiperparámetros:** `RandomizedSearchCV` / `GridSearchCV`.  
- **Métricas costo-sensibles:** AUC-PR; análisis de umbral (curva costo–beneficio).  
- **Validación:** *k-fold CV* para robustez.  
- **Nuevas features:** estacionalidad, variación de cargos, tickets de soporte, morosidad.  
- **Balanceo:** comparar **SMOTE** vs. `class_weight`.

---

## 🖼️ Visualizaciones sugeridas

- Comparativa de métricas por modelo (barras).  
- Matrices de confusión normalizadas.  
- Importancia de variables (RF).  
- Coeficientes ordenados (LR).  
- Boxplots dirigidos (tenure vs. churn; total charges vs. churn).  

---

## 📄 Licencia
Este proyecto se distribuye con fines educativos. Ajustar según políticas del curso/empresa.

---


