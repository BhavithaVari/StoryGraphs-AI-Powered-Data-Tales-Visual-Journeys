
# StoryGraphs: AI-Powered Data Tales & Visual Journeys

## Project Overview
This Streamlit application, **StoryGraphs: AI-Powered Data Tales & Visual Journeys**, offers instant data analysis and executive-style reporting. Users can upload CSV or Excel datasets to receive automated EDA, AI-driven text summaries, data visualizations, and downloadable PDF insights—all in the browser.

## Concepts Covered
- Interactive EDA (descriptive statistics, null value analysis, categorical counts)
- Transformer-based AI summarization (BART-large-cnn via Hugging Face)
- Automated Data Visualizations (histograms, boxplots, heatmaps, countplots)
- PDF Report Generation including insights and all plots
- Data Preprocessing (handles CSV/Excel upload, type detection)
- Robust error handling and session management

## Dataset
Upload your own CSV or Excel file to explore. The application automatically identifies numerical and categorical columns, then generates summaries, visualizations, and text insights based on uploaded data.

## How to Run
1. Install Python 3 and the following libraries: `streamlit`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `fpdf`, `transformers`.
2. Execute from the command line: streamlit run "path/to/app.py"
(Quotes are required if your path contains spaces.)
3. Upload a dataset in the sidebar to begin analysis. Use the provided tabs for EDA, insights, visualizations, and PDF report download.

## Results Summary
StoryGraphs produces:
- Data summary tables (describe, nulls, value counts)
- AI-generated insights using BART summarizer
- Visualizations: Histogram, Boxplot, Correlation Heatmap, Categorical Countplots (limited to top 5 for clarity)
- Executive PDF report download with both textual and graphical summaries
## Structure
- /code - helper Python files
- /dashboard - Streamlit app
- /report - sample generated report (PDF)
- /assets - saved plots used in the report
- /sample_data.csv - bundled sample CSV used for demo
- README.md - this file

# Imarticus Data Science Internship - Assessment (Sample solution)


## Author
Created by Bhavitha


