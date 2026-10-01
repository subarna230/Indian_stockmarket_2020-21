# Indian Stock Market Analysis During the COVID-19 Pandemic

## Assignment by Subarna Mohanta

### Data Analysis and Visualization Using Python

---

## 1. Project Overview

This project analyses the behaviour of the Indian stock market during the COVID-19 pandemic period using Python-based data analysis and visualization techniques.

The project uses historical stock-market data covering the period from **January 2020 to November 2021**, allowing the study of stock-price movements, returns, volatility, risk and market relationships during a period of significant market uncertainty.

The analysis is presented through an interactive dashboard developed using **Plotly and Dash**.

---

## 2. Project Objectives

The main objectives of this project are:

- To analyse historical Indian stock-market data during the COVID-19 pandemic period.
- To examine stock-price movements and daily returns.
- To compare stock performance across different years.
- To identify stocks with high and low returns.
- To analyse stock volatility and risk.
- To study maximum drawdowns from historical peaks.
- To examine the relationship between risk and return.
- To analyse correlations between stock returns.
- To develop an interactive stock-market analytics dashboard.

---

## 3. Dataset

The project uses historical stock-market data containing daily closing prices for a large number of Indian stocks.

### Dataset Information

- **Total observations:** 698,519
- **Stock symbols:** 1,906
- **Time period:** January 2020 – November 2021
- **Main variables:**
  - `Symbol`
  - `Date`
  - `Closing_Price`
  - `Daily_Return`

The dataset was transformed from a wide format into a structured format suitable for analysis and visualization.

---

## 4. COVID-19 Pandemic Context

The selected period covers the major COVID-19 pandemic phase in India.

The analysis focuses on how Indian stocks behaved during this period of considerable economic and market uncertainty.

The project does not attempt to establish direct causation between individual COVID-19 events and specific stock movements. Instead, it provides a historical analysis of market behaviour during the pandemic period.

---

## 5. Technologies Used

The project was developed using:

- **Python**
- **Pandas**
- **NumPy**
- **Plotly**
- **Dash**
- **Jupyter Notebook**

---

## 6. Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the historical stock dataset.
2. Converted dates into proper datetime format.
3. Converted closing prices and returns into numerical values.
4. Removed invalid or missing observations where required.
5. Removed duplicate stock-date records.
6. Sorted observations by stock and date.
7. Calculated daily returns.
8. Created year and month-based variables.
9. Calculated moving averages.
10. Calculated running maximum prices and maximum drawdowns.

---

## 7. Exploratory Data Analysis

The project performs several forms of exploratory analysis, including:

### Stock Performance

- Annual stock returns
- Top-performing stocks
- Lowest-performing stocks
- Average stock prices
- Price trends

### Risk Analysis

- Daily-return volatility
- Stock-level volatility
- Risk versus return
- Maximum drawdown

### Trading Behaviour

- Positive trading days
- Negative trading days
- Trading-day consistency

### Correlation Analysis

The project analyses the correlation between daily returns of selected stocks to understand how different securities moved in relation to one another.

---

## 8. Visualizations

The project includes several visualizations, including:

- Line charts
- Bar charts
- Boxplots
- Pie charts
- Scatter plots
- Correlation heatmaps
- Moving-average charts
- Risk-return plots
- Drawdown charts

These visualizations help identify patterns and differences in stock-market behaviour during the selected period.

---

## 9. Interactive Dashboard

An interactive dashboard was developed using **Dash and Plotly**.

### Dashboard Features

#### Market Overview

Provides a summary of:

- Total stocks
- Number of trading observations
- Average closing price
- Average volatility
- Dataset time period

#### Interactive Stock Explorer

Users can select an individual stock and view:

- Historical closing price
- 7-day moving average
- 30-day moving average
- Daily returns
- Positive and negative trading-day distribution

#### Performance Analytics

Users can select a year and explore:

- Top 10 performing stocks
- Bottom 10 performing stocks
- Daily return distribution

#### Risk Analytics

The dashboard provides:

- Risk versus return analysis
- Most volatile stocks
- Maximum drawdown analysis

#### Correlation Intelligence

A correlation heatmap allows users to examine relationships between stock returns.

---

## 10. Key Findings

The analysis demonstrates substantial variation in stock-market behaviour during the selected COVID-19 pandemic period.

Some major observations include:

- Stock returns varied considerably across different securities.
- Different stocks experienced significantly different levels of volatility.
- Higher returns were sometimes associated with higher levels of risk.
- Several stocks experienced substantial declines from previous historical peaks.
- The market showed considerable differences in positive and negative trading-day consistency across stocks.
- Correlation analysis showed that stocks did not all move in the same way throughout the period.

The analysis therefore highlights the importance of considering both **return and risk** when studying stock-market behaviour.

---

## 11. Project Structure

```text
Indian_Stock_Market_Analysis/
│
├── data/
│   └── All_Stocks_Data_Cleaned.csv
│
├── assets/
│   └── style.css
│
├── reports/
│   └── charts/
│
├── Indian_Stock_Market_Analysis.ipynb
├── app.py
├── requirements.txt
└── README.md
