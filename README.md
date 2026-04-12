# Heart Disease Prediction

Predicting the presence of heart disease from clinical data using three machine learning models. The dataset comes from Kaggle and combines records from 5 different hospital sources — 918 patients total, with 11 features covering demographics, blood tests, and ECG readings.

---

## What I built

Compared three models — Decision Tree, Random Forest, and Gradient Boosting — across five objectives: feature importance, correlation analysis, patient stratification by demographics, a Random Forest model for predicting fasting blood sugar, and a final comparative analysis with ROC curves.

The goal wasn't just to get a high accuracy number. I wanted to understand *which clinical features actually matter* and whether the model's decisions make sense from a medical standpoint. A model that's accurate but uninterpretable isn't useful to a clinician.

---

## Results

| Model | Accuracy |
|---|---|
| Decision Tree | 78.8% |
| Random Forest | 87.5% |
| **Gradient Boosting** | **88.6%** |

Gradient Boosting came out on top. The gap between Decision Tree and the two ensemble methods is significant — single trees overfit on this dataset size (918 rows), while bagging and boosting both reduce that variance substantially.

---

## What actually drives heart disease risk

From feature importance analysis on the Decision Tree:

1. **ST_Slope (39.9%)** — the direction of the ST segment on an ECG. This was by far the most predictive feature, which aligns with clinical literature. A flat or downward ST slope is strongly associated with cardiac ischaemia.
2. **Cholesterol (12.3%)** — as expected, elevated cholesterol is a key risk factor.
3. **MaxHR (8.4%)** — maximum heart rate achieved. Lower max HR during exercise is associated with cardiac problems.
4. **ChestPainType (8.1%)** — asymptomatic chest pain (ASY) turned out to be more predictive of disease than typical angina, which is counterintuitive but well-documented in clinical research.

ExerciseAngina had a correlation of 0.494 with HeartDisease — statistically significant with p < 0.0001.

---

## Patient stratification findings

- Males in this dataset show significantly higher rates of heart disease than females
- Risk peaks in the 51–60 age group — 222 cases vs 159 non-cases in that band
- Patients with exercise-induced angina have substantially higher disease rates than those without — this was the clearest demographic split in the data

---

## Dataset

[Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) from Kaggle. Combines the Cleveland, Hungarian, Switzerland, Long Beach VA, and Stalog heart datasets — 918 observations after removing 272 duplicates from the original 1190.

Features: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope.

---

## How to run it

```bash
git clone https://github.com/Rhiya22/Heart_Disease_Prediction.git
cd Heart_Disease_Prediction
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

Place `heart.csv` in the same folder as the notebook, then open `Heart_Disease_Prediction.ipynb` and run all cells.

---

## Stack

Python, scikit-learn, pandas, numpy, matplotlib, seaborn, scipy
