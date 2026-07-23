# Women's Health Indicator Analysis (2011–2013)

## Overview
This is my individual contribution analyzing CDC BRFSS data on Women's Health indicators from 2011 to 2013 using machine learning.

## What My Code Does
1. Filters raw CDC data specifically for the `Women's Health` class and years 2011–2013.
2. Cleans missing values in `Data_value` and `Sample_Size`.
3. Trains a **Linear Regression** and a **Random Forest Regressor** to predict prevalence values.
4. Performs a fairness check measuring model error across demographic categories (`Break_Out_Category`).

## Results
* **Random Forest** achieved lower error (MAE) compared to Linear Regression.
* Fairness checks showed slightly higher error variance in smaller demographic sample groups.

## How to Run
Open `womens_health_analysis.ipynb` in Google Colab or Jupyter Notebook and run all cells.