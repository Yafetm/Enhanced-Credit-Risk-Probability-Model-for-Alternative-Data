import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Credit Risk Dashboard')
df = pd.read_csv('data/processed/processed.csv')
st.subheader('Processed Data Preview')
st.dataframe(df.head())
st.subheader('SHAP Summary Plot')
st.image('shap_summary.png')
st.subheader('Monetary Distribution')
fig, ax = plt.subplots()
sns.histplot(df['Monetary'], ax=ax)
st.pyplot(fig)