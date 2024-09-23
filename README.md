# Portfolio-Optimization-and-Risk-Management-using-Financial-Analytics-Models

# Project Overview
This project explores various portfolio optimization models using historical data from 17 industry portfolios (sourced from Ken French’s data library) to help balance the trade-off between risk and return. The focus is on three main strategies:

# Mean-Variance Optimization: A strategy that aims to maximize return for a given level of risk.
Minimum Variance Optimization: A strategy that focuses on minimizing portfolio risk without targeting a specific return.
Naive Portfolio Strategy: An equally weighted portfolio that serves as a benchmark for comparison.
The models were evaluated across three estimation windows (12, 36, and 72 months) to assess how different time periods affect portfolio performance.

# Objectives
Develop and implement portfolio optimization models.
Analyze performance across different time horizons to assess the risk-return trade-off.
Compare model results to identify optimal strategies for various investor risk appetites.

# Methodology
The project was developed using Python, leveraging libraries such as:

pandas for data manipulation,
numpy for numerical operations,
matplotlib for data visualization,
scipy for optimization techniques.

# Data Preprocessing
Historical return data was cleaned and normalized to ensure accurate model inputs.
The dataset spans from 1926 to 2024, providing extensive historical data for robust analysis.

# Model Implementation
Mean-Variance Optimization: This model calculates the portfolio that maximizes return for a given level of risk by using mean return and covariance matrices.

Achieved a mean return of 1.04% with a standard deviation of 4.86% using a 36-month window.
Minimum Variance Optimization: This model minimizes risk without regard to return.

The 72-month window yielded the lowest risk with a standard deviation of 3.62%, making it suitable for risk-averse investors.
Naive Portfolio Strategy: This equally weighted approach provided a mean return of 1.13% with a higher standard deviation of 5.31%, reflecting its higher risk and volatility.

# Comparative Analysis
The models were compared across different time windows (12, 36, and 72 months):

12-month window: More responsive to recent market changes but with increased volatility.
36-month window: Balanced risk and return, making it suitable for moderate-risk investors.
72-month window: Smoothed out short-term market fluctuations, resulting in more stable returns.

# Results
The Mean-Variance model (36-month window) provided the highest risk-adjusted return.
The Minimum Variance model (72-month window) minimized risk, ideal for conservative investors.
The Naive Portfolio strategy, while providing the highest return, carried the most risk and volatility.

# Visualizations
Key insights from the models were visualized using scatter plots to compare the risk-return profiles of each strategy. These visualizations highlighted the performance of each model, providing clear insights into the trade-offs between risk and return.

# Key Takeaways
Mean-Variance Optimization is ideal for investors looking for a balance between return and risk.
Minimum Variance Optimization is best suited for investors prioritizing risk minimization.
Naive Portfolio offers higher returns but with significantly increased volatility.

# Technologies Used
Python (pandas, numpy, matplotlib, scipy)
Portfolio Optimization
Risk Management
Data Visualization
Quantitative Finance

# Conclusion
This project demonstrates how different portfolio optimization strategies perform across varying estimation windows. It provides practical insights into how investors can align their strategies with their risk tolerance and investment goals.

# How to Use
Clone this repository and run the Python scripts to:
Perform portfolio optimization using different strategies.
Visualize and analyze the performance of portfolios over different time horizons.
