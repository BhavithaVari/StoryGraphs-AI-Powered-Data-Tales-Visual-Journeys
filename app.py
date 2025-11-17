import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import io
import os
from transformers import pipeline

# App config and title
st.set_page_config(page_title="StoryGraphs: AI-Powered Data Tales & Visual Journeys", layout="wide")
st.title("📈 StoryGraphs: AI-Powered Data Tales & Visual Journeys")

# Hugging Face summarizer pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def remove_non_latin1(text):
    return text.encode('latin-1', errors='ignore').decode('latin-1')

def generate_insight(text):
    max_chunk_size = 1000
    text_chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    summary = ""
    for chunk in text_chunks:
        result = summarizer(chunk, max_length=300, min_length=50, do_sample=False)
        summary += result[0]['summary_text'] + " "
    return summary.strip()

def create_pdf_report(summary, figs, filename="executive_summary.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # First page: Summary text only
    pdf.add_page()
    pdf.set_fill_color(30, 144, 255)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 15, "Executive Summary Report", ln=1, align="C", fill=True)
    pdf.ln(5)

    pdf.set_fill_color(240, 248, 255)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "", 11)
    max_summary_length = 1800
    pdf.multi_cell(0, 8, summary[:max_summary_length], border=1, fill=True)

    # Figures: 2 per page side by side
    margin = 15
    page_width = 210
    usable_width = page_width - 2 * margin
    img_width = (usable_width - 8) / 2
    img_height = 90

    for i in range(0, len(figs), 2):
        pdf.add_page()
        y_start = margin
        title_height = 7

        # Left figure + title
        pdf.set_font("Arial", "B", 14)
        pdf.set_xy(margin, y_start)
        pdf.cell(img_width, title_height, f"Visualization {i+1}", align="L")
        y_img = y_start + title_height + 2

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile1:
            figs[i].savefig(tmpfile1.name, format="png", bbox_inches="tight")
            img1_path = tmpfile1.name

        pdf.image(img1_path, x=margin, y=y_img, w=img_width, h=img_height)
        os.remove(img1_path)

        # Right figure + title (if exists)
        if i + 1 < len(figs):
            pdf.set_font("Arial", "B", 14)
            pdf.set_xy(margin + img_width + 8, y_start)
            pdf.cell(img_width, title_height, f"Visualization {i+2}", align="L")
            y_img_right = y_start + title_height + 2

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile2:
                figs[i+1].savefig(tmpfile2.name, format="png", bbox_inches="tight")
                img2_path = tmpfile2.name

            pdf.image(img2_path, x=margin + img_width + 8, y=y_img_right, w=img_width, h=img_height)
            os.remove(img2_path)

    output = io.BytesIO(pdf.output(dest="S").encode("latin-1"))
    return output

def generate_nlp_insight(df):
    st.subheader("🧠 AI-Generated Insights")
    try:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        st.markdown(f"✅ Dataset shape: **{df.shape[0]} rows x {df.shape[1]} columns**")
        st.markdown(f"🔹 Numerical columns: **{', '.join(num_cols) if num_cols else 'None'}**")
        st.markdown(f"🔹 Categorical columns: **{', '.join(cat_cols) if cat_cols else 'None'}**")

        summary_lines = []
        summary_lines.append("Describe statistics:")
        summary_lines.append(df.describe(include="all").fillna('').to_string())
        if num_cols:
            summary_lines.append("\nMedian values:")
            summary_lines.append(df.median(numeric_only=True).to_string())
            summary_lines.append("\nVariance values:")
            summary_lines.append(df.var(numeric_only=True).to_string())
        summary_lines.append("\nNull counts:")
        summary_lines.append(df.isnull().sum().to_string())

        if cat_cols:
            summary_lines.append("\nCategory distributions:")
            for col in cat_cols:
                summary_lines.append(f"\n{col}:")
                value_counts = df[col].value_counts()
                summary_lines.append(value_counts.to_string())

        summary_text = "\n".join(summary_lines)[:2000]
        short_summary = f"Data shape: {df.shape}, columns: {list(df.columns)}"

        if st.button("🤖 Run AI Summarizer"):
            with st.spinner("Generating AI insights..."):
                insights1 = generate_insight(summary_text)
                insights2 = generate_insight(short_summary)
                st.session_state['insights1'] = insights1
                st.session_state['insights2'] = insights2
                st.success(f"💡 {insights1}")
                st.info(f"✨ Additional Insight: {insights2}")
        elif 'insights1' in st.session_state and 'insights2' in st.session_state:
            st.success(f"💡 {st.session_state['insights1']}")
            st.info(f"✨ Additional Insight: {st.session_state['insights2']}")
    except Exception as e:
        st.error(f"⚠️ Failed to generate insights: {e}")

uploaded_file = st.sidebar.file_uploader("⬆️ Upload CSV or Excel File", type=["csv", "xls", "xlsx"])
if not uploaded_file:
    st.info("⚡ Upload your CSV or Excel dataset for instant data analysis, insights, charts, and a downloadable report.")

if uploaded_file:
    file_ext = uploaded_file.name.split('.')[-1].lower()
    if file_ext == "csv":
        df = pd.read_csv(uploaded_file)
    elif file_ext in ["xls", "xlsx"]:
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file type")
        st.stop()

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if st.checkbox("👀 Show Data Preview (Head)"):
        st.subheader("🗂 Data Preview")
        st.dataframe(df.head())

    tabs = st.tabs(["📋 Exploratory Data Analysis", "🧠 AI-Generated Insights", "📊 Visualizations", "📄 Report"])

    with tabs[0]:
        st.markdown("### 🔍 Exploratory Data Analysis")
        st.subheader("Numerical Data Summary")
        st.dataframe(df.describe())
        st.subheader("Null values per column:")
        nulls_df = pd.DataFrame(df.isnull().sum(), columns=["Null Count"])
        nulls_df.index.name = "Column"
        st.dataframe(nulls_df)
        st.subheader("Categorical Value Counts")
        for col in cat_cols:
            with st.expander(f"{col} Value Counts"):
                st.dataframe(df[col].value_counts())

    with tabs[1]:
        generate_nlp_insight(df)

    with tabs[2]:
        st.markdown("### 📊 Automated Data Visualizations")
        figs = []
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if num_cols:
            # Histogram
            fig_hist, ax_hist = plt.subplots(figsize=(6,4))
            df[num_cols].hist(ax=ax_hist)
            ax_hist.set_title("Histogram of Numerical Features")
            st.pyplot(fig_hist)
            figs.append(fig_hist)
            plt.close(fig_hist)

            # Boxplot
            fig_box, ax_box = plt.subplots(figsize=(6,4))
            sns.boxplot(data=df[num_cols], ax=ax_box)
            ax_box.set_title("Boxplot of Numerical Features")
            st.pyplot(fig_box)
            figs.append(fig_box)
            plt.close(fig_box)

            # Correlation heatmap if more than 1 numeric column
            if len(num_cols) > 1:
                fig_corr, ax_corr = plt.subplots(figsize=(8,6))
                corr_mat = df[num_cols].corr()
                sns.heatmap(corr_mat, annot=True, cmap="viridis", ax=ax_corr)
                ax_corr.set_title("Numeric Feature Correlation")
                st.pyplot(fig_corr)
                figs.append(fig_corr)
                plt.close(fig_corr)

        # Categorical countplots (limit to 5 for readability)
        for col in cat_cols[:5]:
            fig_cat, ax_cat = plt.subplots(figsize=(6,4))
            sns.countplot(data=df, x=col, order=df[col].value_counts().index, ax=ax_cat)
            ax_cat.set_title(f"Countplot of {col}")
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig_cat)
            figs.append(fig_cat)
            plt.close(fig_cat)

    with tabs[3]:
        st.markdown("### 📄 Download Executive Summary Report")
        insights1 = st.session_state.get('insights1', None)
        insights2 = st.session_state.get('insights2', None)
        if insights1 and insights2 and figs:
            eda_summary = df.describe().to_string()
            # Summary of category counts only, no detailed counts
            cat_value_counts_str = "\n\nCategory Counts Summary:\n"
            for col in cat_cols:
                unique_count = df[col].nunique()
                cat_value_counts_str += f"{col}: {unique_count} unique categories\n"

            full_summary = f"Exploratory Data Analysis Summary:\n{eda_summary}{cat_value_counts_str}\n\nAI Insight 1:\n{insights1}"
            clean_summary = remove_non_latin1(full_summary)
            pdf_bytes = create_pdf_report(clean_summary, figs)
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name="executive_summary.pdf",
                mime="application/pdf"
            )
        else:
            st.warning("⚠️ Please generate insights and visualizations first.")
