# -*- coding: utf-8 -*-
"""
Created on Fri May 30 01:20:41 2025

@author: Administrator

Sheet Link: https://docs.google.com/spreadsheets/d/1OBVjTQ2KjLqxBhSQ2l203RBzcukK7dZCixmx0zYbXfo/edit?usp=sharing

"""

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("depression_dataset_with_smote.csv")

df = load_data()

st.title("Depression Dataset Visualization")

# Bar Chart: Distribution of Depression
st.subheader("Distribution of Depression")
dep_counts = df['target/Depression'].value_counts().reset_index()
dep_counts.columns = ['Depression', 'Count']
fig_bar = px.bar(dep_counts, x='Depression', y='Count', color='Depression', title="Depression Counts")
st.plotly_chart(fig_bar)
st.markdown("This bar chart shows the number of individuals with and without depression in the dataset.")

# Pie Chart: Suicidal Thoughts
st.subheader("Suicidal Thoughts Distribution")
suicidal_counts = df['Suicidal thoughts'].value_counts().reset_index()
suicidal_counts.columns = ['Suicidal thoughts', 'Count']
fig_pie = px.pie(suicidal_counts, values='Count', names='Suicidal thoughts', title="Suicidal Thoughts")
st.plotly_chart(fig_pie)
st.markdown("This pie chart illustrates the proportion of individuals who reported having suicidal thoughts versus those who did not.")

# Histogram: Sleep Duration
st.subheader("Histogram of Sleep Duration")
fig_hist = px.histogram(df, x="Sleep Duration", nbins=10, title="Distribution of Sleep Duration")
st.plotly_chart(fig_hist)
st.markdown("This histogram displays how sleep duration is distributed among individuals in the dataset.")

# Box Plot: CGPA vs Depression
st.subheader("Box Plot of CGPA by Depression Status")
fig_box = px.box(df, x="target/Depression", y="CGPA", title="CGPA by Depression Status")
st.plotly_chart(fig_box)
st.markdown("This box plot compares CGPA distributions between those with and without depression.")

# Correlation Heatmap
st.subheader("Correlation Heatmap")
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
st.pyplot(plt.gcf())
st.markdown("This heatmap shows the correlation coefficients between all numerical features in the dataset.")

# Scatter Plot: Study Satisfaction vs Academic Pressure
st.subheader("Scatter Plot: Study Satisfaction vs Academic Pressure")
fig_scatter = alt.Chart(df).mark_circle(size=60).encode(
    x='Study Satisfaction',
    y='Academic Pressure',
    color='target/Depression:N',
    tooltip=['CGPA', 'Sleep Duration']
).interactive()
st.altair_chart(fig_scatter, use_container_width=True)
st.markdown("This scatter plot displays the relationship between study satisfaction and academic pressure, colored by depression status.")

st.success("Visualization complete.")
