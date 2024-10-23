import time
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import DatasetConfig
from src.load_dataset import load_df

def visualize_full_correlation_heatmap(df):
    sns.set_context("paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(df.corr(numeric_only=True), cmap="YlGnBu", annot=True, ax=ax)
    plt.title('Correlation Heatmap for All Features')
    plt.tight_layout()
    return fig

def visualize_selected_correlation_heatmap(df):
    new_df = df[['TV', 'Radio', 'Social Media', 'Sales']]
    sns.set_context("paper", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(new_df.corr(numeric_only=True), cmap="YlGnBu", annot=True, ax=ax)
    plt.title('Correlation Heatmap for TV, Radio, Social Media, and Sales')
    plt.tight_layout()
    return fig

def visualize_marketing_sales_pairplot(df):
    sns.set_context("notebook", font_scale=0.5)
    fig = sns.pairplot(
        data=df,
        x_vars=['TV', 'Radio', 'Social Media'],
        y_vars='Sales',
        height=2,
        kind='reg'
    )
    plt.title('Pairplot for TV, Radio, Social Media vs Sales')
    plt.tight_layout()
    return fig

def visualize_influencer_sales_pairplot(df):
    sns.set_context("notebook", font_scale=0.9)
    fig = sns.pairplot(
        data=df,
        x_vars=['Influencer_Mega', 'Influencer_Micro', 'Influencer_Nano', 'Influencer_Macro'],
        y_vars='Sales',
        height=3,
        kind='reg'
    )
    plt.title('Pairplot for Influencers vs Sales')
    plt.tight_layout()
    return fig

def plot_figure():
    df = load_df(DatasetConfig.DATASET_PATH)

    st.subheader("Data Analysis and Visualization")
    with st.spinner('Processing...'):
        time.sleep(2)

        full_corr_fig = visualize_full_correlation_heatmap(df)
        selected_corr_fig = visualize_selected_correlation_heatmap(df)

        marketing_pairplot_fig = visualize_marketing_sales_pairplot(df)
        influencer_pairplot_fig = visualize_influencer_sales_pairplot(df)

    st.markdown("""
    #### Heatmap Analysis:
    - **The full correlation heatmap** shows the relationships among all numerical variables.
    - **The selected correlation heatmap** focuses on the correlation between TV, Radio, Social Media, and Sales.

    #### Pairplot Analysis:
    - **The marketing pairplot** compares Sales against TV, Radio, and Social Media expenditures.
    - **The influencer pairplot** shows how different types of influencers relate to Sales.
    """)

    cols = st.columns(2)
    
    with cols[0]:
        st.pyplot(full_corr_fig)
    with cols[1]:
        st.pyplot(selected_corr_fig)
    st.pyplot(marketing_pairplot_fig)
    st.pyplot(influencer_pairplot_fig)