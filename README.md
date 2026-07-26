# Women's Health Indicator Analysis (2011–2013)

## Overview
This repository contains my individual capstone analysis on CDC BRFSS Women's Health data covering 2011 through 2013.

## Project Deliverables & Rubric Requirements
* **Data Cleaning (10%):** Processed and cleaned missing values across 5 key variables (`Data_value`, `Sample_Size`, `Year`, `Break_Out_Category`, `Locationabbr`).
* **Exploratory Data Analysis (15%):** Generated summary statistics and distributions for 5 core feature variables.
* **Data Visualization (10%):** Built 5 distinct charts analyzing health indicator rates by year, state, demographic breakdown, and sample size.
* **Machine Learning Models (15%):** Evaluated performance across 5 regression algorithms (`Linear Regression`, `Ridge`, `K-Nearest Neighbors`, `Random Forest`, and `Gradient Boosting`).

## Key Findings
* The **Gradient Boosting** and **Random Forest** models achieved the lowest Mean Absolute Error (MAE) when predicting indicator prevalence rates.
* Demographic groups showed stable reporting trends across the 2011–2013 observation window.

## How to Run
1. Open `womens_health_analysis.ipynb` in Google Colab or Jupyter Notebook.
2. Execute all cells sequentially. The raw dataset loads directly from the CDC web database API.
