# CCPP Power Output Prediction

An end-to-end Machine Learning project that predicts electrical power output (PE) of a Combined Cycle Power Plant using environmental and operational parameters.

**Live Demo:** https://ccpp-power-predictor.streamlit.app/

## 1. Problem Statement

The objective is to predict the electrical power output of a Combined Cycle Power Plant using four input parameters:

- Ambient Temperature (AT)
- Exhaust Vacuum (V)
- Ambient Pressure (AP)
- Relative Humidity (RH)

This is a supervised regression problem.

## 2. Dataset

Dataset: Combined Cycle Power Plant Dataset (UCI Machine Learning Repository).

- Original observations: 9,568
- Features: 4
- Target: 1 (PE)
- Missing values: 0
- Duplicate rows removed: 41
- Final observations: 9,527

## 3. Exploratory Data Analysis

The EDA process includes:

- Dataset structure and data types.
- Missing-value analysis.
- Duplicate detection and removal.
- Descriptive statistics.
- Univariate and bivariate analysis.
- Correlation heatmap.
- Distribution analysis.

## 4. Feature Engineering and Preprocessing

- Separated independent variables and target variable.
- Removed duplicate observations.
- Performed an 80:20 train-test split.
- Used StandardScaler for Linear Regression.
- Used Scikit-learn Pipelines to organise preprocessing and modelling.

Tree-based models were trained without feature scaling.

## 5. Model Training and Comparison

Three regression algorithms were evaluated:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

Models were compared using MAE, RMSE and R².

Five-fold cross-validation was implemented for model evaluation. Random Forest hyperparameters were further optimised using GridSearchCV.

### Hyperparameter Tuning

The tuned Random Forest model uses:

- n_estimators: 300
- max_depth: 20
- min_samples_split: 2
- min_samples_leaf: 1

## 6. Final Model Evaluation

The tuned Random Forest was selected as the final model.

| Metric | Test Result |
|---|---:|
| MAE | 2.41 MW |
| RMSE | 3.44 MW |
| R² Score | 0.9598 |

The model explains approximately 96% of the variance in electrical power output on the held-out test set.

## 7. Bias and Variance Analysis

Training and testing RMSE were compared to investigate overfitting.

The tuned Random Forest achieved:

- Training RMSE: 1.24 MW
- Testing RMSE: 3.44 MW

The difference indicates some overfitting and higher variance.

Residual analysis shows that most errors are distributed around zero, although some large outliers remain.

## 8. Deployment

The project is deployed using Streamlit Community Cloud.

Users can enter the four input parameters and obtain predicted electrical power output.

Docker is used to containerise the application.

## 9. CI/CD

GitHub Actions is configured to automatically run the project validation workflow when changes are pushed to the repository.

## 10. Project Structure

```text
CCPP-ML-Project/
│
├── data/
│   └── ccpp.csv
│
├── models/
│   ├── ccpp_model.joblib
│   └── model_metrics.json
│
├── notebooks/
│   └── 01_EDA.ipynb
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── app.py
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

## 11. Running Locally

Clone the repository:

```bash
git clone https://github.com/arthgoel2007-sys/CCPP-ML-Project.git
cd CCPP-ML-Project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## 12. Running with Docker

Build the Docker image:

```bash
docker build -t ccpp-power-predictor .
```

Run the container:

```bash
docker run -p 8501:8501 ccpp-power-predictor
```

Open:

```text
http://localhost:8501
```

## 13. Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib
- Docker
- Git and GitHub
- GitHub Actions

---

**Project:** Combined Cycle Power Plant Electrical Power Prediction

**Model:** Tuned Random Forest Regression