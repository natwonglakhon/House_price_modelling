# California Housing Price Prediction

A machine learning project that predicts California median house prices using linear regression (SGDRegressor) with scikit-learn.

---

## Overview

This project covers a full ML pipeline, starting from raw data exploration and cleaning, through to model training, evaluation, and visualisation, using the California Housing dataset.

---

## Repository Structure

```
california-housing/
├── README.md
├── California_Housing.ipynb
├── housing.csv
├── requirements.txt
└── images/
    ├── median_house_value_distribution.png
    ├── median_house_value_distribution_filtered.png
    ├── data_vs_price.png
    ├── data_distribution.png
    └── Test_vs_model.png
```

---

## Dataset

- **Source:** [Kaggle - California Housing Prices](https://www.kaggle.com/datasets/nalisha/california-housing-prices-dataset-clean-and-ml/data)
- **Shape:** 20,640 rows x 10 features (before filtering)
- **Target:** `median_house_value`
- **Features:** `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `ocean_proximity`

---

## Methods

### 1. Data Cleaning
- Filled missing values in `total_bedrooms` with the column median
- Filtered out capped values near $500,000, which is a known artefact in this dataset that hurts model performance

### 2. Feature Engineering
- Applied one-hot encoding (`get_dummies`) on the categorical `ocean_proximity` column, turning 5 categories into 4 dummy variables using `drop_first=True`

### 3. Train/Test Split
- Split the data 70% training and 30% testing using `train_test_split` with `random_state=42`

### 4. Feature Scaling
- Used `StandardScaler` (Z-score normalisation: mean=0, std=1)
- The scaler was fitted only on training data to avoid data leakage into the test set

### 5. Model
- Trained a **SGDRegressor** (Stochastic Gradient Descent linear regression) with `max_iter=2000`

---

## Results

| Metric | Score |
|--------|-------|
| R2 Score | ~0.615 |

The model lands at around 61.5% R2. This is a reasonable result for linear regression on this dataset, as the relationship between features and house price is not purely linear. Tree-based models like Random Forest typically do better here.

### Visualisations

**Target distribution (after filtering)**

![Price Distribution](images/median_house_value_distribution_filtered.png)

**Feature vs Price**

![Data vs Price](images/data_vs_price.png)

**Test data vs Model Prediction**

![Test vs Prediction](images/Test_vs_model.png)

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/natwonglakhon/california-housing.git
cd california-housing
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch the notebook:
```bash
jupyter notebook California_Housing.ipynb
```

---

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
jupyter
```

---

## Lessons Learned

- Always split your data before evaluating. Scoring on training data gives a misleadingly good result.
- Fit the scaler on training data only. Fitting on the full dataset leaks test information into the model.
- Data quality has a real impact. Removing the capped $500k values gave a noticeable improvement in R2.
- Linear regression has limits. When the underlying relationships are non-linear, the model will hit a ceiling no matter how well you tune it.

---

## Future Improvements

- Try Random Forest or XGBoost, which typically reach above 0.80 R2 on this dataset
- Explore additional feature engineering, such as rooms per household or bedrooms per room
- Add a correlation heatmap to better understand which features actually matter
- Run hyperparameter tuning with `GridSearchCV`

---

## License

This project is open source and available under the [MIT License](LICENSE).