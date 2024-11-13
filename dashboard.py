import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from statsmodels.tsa.arima.model import ARIMA
from mlxtend.frequent_patterns import apriori, association_rules

# Load data (replace with your actual dataset path)
data = pd.read_csv('realistic_e_commerce_sales_data.csv', parse_dates=['Order Date'])

# Dashboard title
st.title("Comprehensive E-commerce Insights Dashboard")

# Sidebar filters
st.sidebar.subheader("Filters")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime(data['Order Date'].min()))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime(data['Order Date'].max()))
selected_categories = st.sidebar.multiselect("Select Product Categories", options=data['Category'].unique(), default=data['Category'].unique())

# Filter data by user inputs
filtered_data = data[(data['Order Date'] >= pd.to_datetime(start_date)) & (data['Order Date'] <= pd.to_datetime(end_date))]
filtered_data = filtered_data[filtered_data['Category'].isin(selected_categories)]

# 1. Dynamic KPI Metrics
st.subheader("Dynamic KPI Metrics")
total_sales = filtered_data['Total Price'].sum()
avg_order_value = filtered_data['Total Price'].mean()
top_customer = filtered_data.groupby('Customer ID')['Total Price'].sum().sort_values(ascending=False).head(1)
st.metric("Total Sales", f"${total_sales:,.2f}")
st.metric("Average Order Value", f"${avg_order_value:,.2f}")
st.metric("Top Customer ID", top_customer.index[0])

# 2. RFM Analysis Summary
st.subheader("RFM Analysis Summary")
rfm = pd.read_csv('rfm_results.csv')  # Load pre-calculated RFM data if available
st.dataframe(rfm.head())

# 3. Customer Segmentation Visualization
st.subheader("Customer Segmentation Clusters")
fig, ax = plt.subplots()
sns.scatterplot(x=rfm['Recency'], y=rfm['Monetary'], hue=rfm['Cluster'], palette='tab10', ax=ax)
ax.set_title('Customer Segmentation Clusters')
st.pyplot(fig)

# 4. Churn Analysis Dashboard
st.subheader("Churn Analysis Dashboard")
churn_rate = 1 - (filtered_data['Customer ID'].nunique() / data['Customer ID'].nunique())
churn_rate_percentage = churn_rate * 100
st.metric("Churn Rate", f"{churn_rate_percentage:.2f}%")

# 5. Predictive Sales Forecasting
st.subheader("Predictive Sales Forecasting")
periods = st.sidebar.slider("Forecast Period (Months)", min_value=1, max_value=12, value=6)
sales_time_series = filtered_data.groupby(filtered_data['Order Date'].dt.to_period('M'))['Total Price'].sum()
sales_time_series.index = sales_time_series.index.to_timestamp()
model = ARIMA(sales_time_series, order=(5, 1, 0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=periods)
fig, ax = plt.subplots(figsize=(14, 6))
sales_time_series.plot(ax=ax, label='Historical Sales')
forecast.plot(ax=ax, label='Forecast', color='red', linestyle='--')
ax.set_title('Sales Forecast for the Next Few Months')
ax.set_xlabel('Date')
ax.set_ylabel('Total Sales')
ax.legend()
st.pyplot(fig)

# 6. Top Customer Identification
st.subheader("Top Customers by Total Sales")
top_customers = filtered_data.groupby('Customer ID')['Total Price'].sum().sort_values(ascending=False).head(10)
st.bar_chart(top_customers)

# 7. Product Performance Drill-Down
st.subheader("Product Performance Drill-Down")
selected_product = st.selectbox("Select a Product to View Performance", options=filtered_data['Product Name'].unique())
product_sales = filtered_data[filtered_data['Product Name'] == selected_product].groupby(filtered_data['Order Date'].dt.to_period('M'))['Total Price'].sum()
fig, ax = plt.subplots()
product_sales.plot(ax=ax, kind='line', marker='o')
ax.set_title(f'Sales Trend for {selected_product}')
ax.set_xlabel('Time (Month)')
ax.set_ylabel('Total Sales')
st.pyplot(fig)

# 8. Retention by Product Category (Customizable)
st.subheader("Retention by Product Category")
category_retention = filtered_data.groupby('Category')['Customer ID'].nunique() / filtered_data['Customer ID'].nunique()
fig, ax = plt.subplots()
sns.barplot(x=category_retention.index, y=category_retention.values, palette='Blues', ax=ax)
ax.set_title('Retention by Product Category')
ax.set_xlabel('Product Category')
ax.set_ylabel('Retention Rate')
st.pyplot(fig)

# 9. CLV Calculator
st.subheader("Customer Lifetime Value (CLV) Calculator")
# Calculate average order value (AOV)
avg_order_value = filtered_data['Total Price'].mean()
# Calculate purchase frequency (average number of purchases per customer)
purchase_frequency = filtered_data['Customer ID'].value_counts().mean()
# Calculate churn rate (using a basic approach for demonstration)
churn_rate = 1 - (filtered_data['Customer ID'].nunique() / data['Customer ID'].nunique())
# Avoid division by zero if churn_rate is 0
if churn_rate == 0:
    churn_rate = 0.001  # Small value to avoid errors and indicate a low churn
# Calculate CLV
clv = avg_order_value * purchase_frequency / churn_rate
# Display CLV
st.metric("Estimated CLV", f"${clv:,.2f}")

# 10. Interactive Correlation Matrix
st.subheader("Interactive Correlation Matrix")
# Select only numeric columns for the correlation matrix
numeric_data = filtered_data.select_dtypes(include=[np.number])
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Correlation Matrix')
st.pyplot(fig)