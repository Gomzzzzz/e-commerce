# E-commerce Customer Insights Dashboard

## Overview
This project is an interactive **E-commerce Customer Insights Dashboard** built using **Streamlit**. It provides comprehensive analyses and visualizations to help stakeholders understand customer behavior, retention trends, and sales performance. This tool supports data-driven decision-making for business improvement.

## Features
The dashboard includes the following key features:
1. **Date Range and Category Filters**: Customize the analysis by filtering data by date range and product category.
2. **Dynamic KPI Metrics**: View essential business metrics like total sales, average order value, and top customer ID.
3. **RFM Analysis Summary**: Recency, Frequency, and Monetary analysis for identifying valuable customers.
4. **Customer Segmentation Visualization**: Clusters of customers visualized based on purchasing behavior.
5. **Churn Analysis Dashboard**: Display churn rate and associated metrics for customer retention.
6. **Predictive Sales Forecasting**: Visualize future sales trends using ARIMA time series forecasting.
7. **Top Customer Identification**: Highlight the top customers by total sales.
8. **Product Performance Drill-Down**: Select products to see detailed sales trends over time.
9. **Retention by Product Category**: Display retention rates by product category with customizable chart types.
10. **Customer Lifetime Value (CLV) Calculator**: Estimate the potential revenue a customer may bring during their lifetime.
11. **Interactive Correlation Matrix**: Analyze relationships between numerical data features.

## Dataset
The dataset used in this project is sourced from [Kaggle's E-commerce Sales Data](https://www.kaggle.com/datasets/refiaozturk/e-commerce-sales?resource=download). Special thanks to the original data contributors for making this project possible.

## Prerequisites
Make sure the following are installed:
- **Python 3.7 or later**
- Required Python libraries:
  bash
  pip install streamlit pandas numpy matplotlib seaborn scikit-learn lifelines mlxtend statsmodels

## Installation and Setup
1. Clone the repository:

bash
Copy code
git clone https://github.com/Gomzzzzz/e-commerce.git

2. cd ecommerce

3. Run the dashboard:

bash
Copy code
streamlit run dashboard.py

4. Access the dashboard: Open the provided local URL (e.g., http://localhost:8501) in your web browser.

## How to Use the Dashboard
Use the Date Range and Category Filters to refine the analysis.
Navigate through each interactive section to explore customer behavior, sales forecasts, and retention trends.
Use the CLV Calculator to estimate customer value and the Interactive Correlation Matrix to explore data relationships.

## Credits
Dataset: Kaggle's E-commerce Sales Data

## License
This project is licensed under the MIT License.
