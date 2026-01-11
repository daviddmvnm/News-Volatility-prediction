# News & Volatility Prediction

Predicting extreme market volatility using semantic features from large-scale financial news.

## Project Structure
```
├── pipelines/          # Data processing (Bronze → Silver → Gold medallion architecture)
├── clustering/         # Two-stage news clustering + LLM-assisted theme labelling
├── modelling/          # EDA, feature engineering, baseline models, hybrid residual framework
├── requirements.txt
```

## Overview

This project tests whether news has short-horizon predictive power for volatility. Key components:

1. **Data Pipeline**: Process 57M news articles down to ~2M relevant financial articles using PySpark
2. **Clustering**: Two-stage K-Means with LLM interpretation to extract 7 thematic clusters (Markets, Macro, Earnings, Energy, Tech, Trade, Geopolitics)
3. **Feature Engineering**: Daily sentiment intensity, activity share, uncertainty, burst features, regime indicators
4. **Modelling**: Hybrid residual framework combining linear persistence (logistic regression) with nonlinear news effects (random forest)

## Key Finding

News effects are **nonlinear and regime-dependent**. Linear models capture volatility persistence but miss news signals. Nonlinear models capture news effects but can't model persistence. The hybrid approach outperforms both.

## Data

- **News**: [Financial News Multisource](https://huggingface.co/datasets/Brianferrell787/financial-news-multisource) (57M articles)
- **Market**: Yahoo Finance via `yfinance`
- **Processed Features**: [Google Drive](https://drive.google.com/drive/folders/1FFkeFcLwR7XuvW-mU_Um4EJlmSUq3AQr)

## Interactive Dashboard

[Live Demo](https://news-volatility-predictiondashboard-zs3egth3qvtkrfzp5cmczj.streamlit.app/) — Enter any ticker and explore model performance.

## Report
- [Project Report (PDF)](https://docs.google.com/document/d/1d13AOZHMbSHORa-LI81_pXSDOy3eiwBfxVOXyzW3Y7w/edit?tab=t.cujvg6q004tp)
