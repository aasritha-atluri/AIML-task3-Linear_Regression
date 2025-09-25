# AIML-task3-Linear_Regression
Implementation of Simple and Multiple Linear Regression on the Housing Price Prediction dataset (AIML Internship Task 3).

## Requirements
- Python 3.x
- Install dependencies before running:

```bash
pip install pandas numpy matplotlib scikit-learn
```
## How to Run
- Clone this repository or download the files.
- Download the dataset Housing.csv.
- Place Housing.csv in the same folder as task3code.py.
- Run the script:

```bash
python task3code.py
```

- The script will print summary info, evaluation metrics (MAE, MSE, R²), coefficients, and show regression plots.

## What I Did

- Imported and preprocessed the dataset.
- Converted categorical variables into numerical using one-hot encoding.
- Split the dataset into training and testing sets.
- Fitted a Linear Regression model using scikit-learn.
- Evaluated performance using **MAE, MSE, and R² score**.
- Plotted regression line for **area vs price**.
- Interpreted coefficients for feature impact.

## Key Insights
- R² indicates how well the independent variables explain the variation in price.
- Higher area and furnishing status positively influence price.
- Location and other categorical variables also affect pricing.

## Tools & Libraries
- **Python**
- **Pandas, NumPy** → Data handling
- **Matplotlib** → Visualization
- **Scikit-learn** → Linear Regression model & metrics

## Files in this Repository
- task3code.py → Code for linear regression
- Housing.csv → Raw dataset (downloaded from Kaggle)
- README.md → Documentation
