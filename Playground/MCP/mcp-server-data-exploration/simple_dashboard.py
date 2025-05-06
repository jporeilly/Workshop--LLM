import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Student Performance Analysis",
    page_icon="📚",
    layout="wide"
)

# Add server configuration
if __name__ == '__main__':
    import os
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'

# Add title and description
st.title("Student Performance Analysis Dashboard")
st.write("This dashboard provides insights into student performance based on various factors.")

# Define function to load data
@st.cache_data()
def load_data():
    try:
        # Try to load from the current directory first
        df = pd.read_csv("Students_Grading_Dataset.csv")
        return df
    except FileNotFoundError:
        try:
            # Try the full path specified
            df = pd.read_csv("C:/mcp-server-data-exploration/data/Students_Grading_Dataset.csv")
            return df
        except FileNotFoundError:
            st.error("Could not find the CSV file. Please check the file path.")
            return None

# Load the data
df = load_data()

if df is not None:
    st.success(f"Data loaded successfully! Found {len(df)} student records.")
    
    # Sidebar for filtering
    st.sidebar.title("Filters")
    
    # Department filter
    departments = ['All'] + sorted(df['Department'].unique().tolist())
    selected_department = st.sidebar.selectbox("Select Department", departments)
    
    # Gender filter
    genders = ['All'] + sorted(df['Gender'].unique().tolist())
    selected_gender = st.sidebar.selectbox("Select Gender", genders)
    
    # Apply filters
    filtered_df = df.copy()
    if selected_department != 'All':
        filtered_df = filtered_df[filtered_df['Department'] == selected_department]
        
    if selected_gender != 'All':
        filtered_df = filtered_df[filtered_df['Gender'] == selected_gender]
    
    # Display filter information
    st.write(f"Showing data for **{len(filtered_df)}** students")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Grade Analysis", "Performance Metrics", "Lifestyle Factors"])
    
    with tab1:
        st.header("Grade Distribution")
        
        # Grade distribution chart
        fig, ax = plt.subplots(figsize=(10, 6))
        grade_counts = filtered_df['Grade'].value_counts().sort_index()
        sns.barplot(x=grade_counts.index, y=grade_counts.values, ax=ax)
        ax.set_title("Grade Distribution")
        ax.set_xlabel("Grade")
        ax.set_ylabel("Number of Students")
        
        # Add count labels on top of bars
        for i, count in enumerate(grade_counts.values):
            ax.text(i, count + 5, str(count), ha='center')
            
        st.pyplot(fig)
        
        # Grade distribution by department
        if len(filtered_df['Department'].unique()) > 1:
            st.subheader("Grade Distribution by Department")
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create cross-tabulation
            grade_dept = pd.crosstab(filtered_df['Department'], filtered_df['Grade'])
            grade_dept.plot(kind='bar', stacked=True, ax=ax)
            
            ax.set_title("Grade Distribution by Department")
            ax.set_xlabel("Department")
            ax.set_ylabel("Count")
            ax.legend(title="Grade")
            
            st.pyplot(fig)
    
    with tab2:
        st.header("Performance Metrics")
        
        # Select metrics to display
        st.subheader("Select Performance Metrics to Compare by Grade")
        cols = st.columns(3)
        
        with cols[0]:
            show_midterm = st.checkbox("Midterm Score", value=True)
        with cols[1]:
            show_final = st.checkbox("Final Score", value=True)
        with cols[2]:
            show_total = st.checkbox("Total Score", value=True)
        
        # Performance by grade
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics_to_show = []
        if show_midterm:
            metrics_to_show.append('Midterm_Score')
        if show_final:
            metrics_to_show.append('Final_Score')
        if show_total:
            metrics_to_show.append('Total_Score')
            
        if metrics_to_show:
            performance_by_grade = filtered_df.groupby('Grade')[metrics_to_show].mean()
            performance_by_grade.plot(kind='bar', ax=ax)
            
            ax.set_title("Average Scores by Grade")
            ax.set_xlabel("Grade")
            ax.set_ylabel("Average Score")
            ax.legend(title="Metric")
            
            st.pyplot(fig)
        else:
            st.info("Please select at least one metric to display")
            
        # Score distributions
        st.subheader("Score Distributions")
        
        selected_score = st.selectbox(
            "Select Score to Visualize",
            options=['Total_Score', 'Midterm_Score', 'Final_Score', 'Assignments_Avg', 'Quizzes_Avg']
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(filtered_df[selected_score], kde=True, ax=ax)
        ax.set_title(f"Distribution of {selected_score}")
        ax.set_xlabel(selected_score)
        ax.set_ylabel("Count")
        
        st.pyplot(fig)
    
    with tab3:
        st.header("Lifestyle Factors and Academic Performance")
        
        # Study hours vs total score
        st.subheader("Study Hours and Performance")
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=filtered_df,
            x='Study_Hours_per_Week',
            y='Total_Score',
            hue='Grade',
            alpha=0.7,
            ax=ax
        )
        
        ax.set_title("Study Hours vs Total Score")
        ax.set_xlabel("Study Hours per Week")
        ax.set_ylabel("Total Score")
        
        st.pyplot(fig)
        
        # Sleep hours impact
        st.subheader("Sleep Hours and Performance")
        
        # Create categories for sleep hours
        filtered_df['Sleep_Category'] = pd.cut(
            filtered_df['Sleep_Hours_per_Night'],
            bins=[3.9, 5.0, 6.0, 7.0, 8.0, 9.1],
            labels=['4-5', '5-6', '6-7', '7-8', '8-9']
        )
        
        sleep_perf = filtered_df.groupby('Sleep_Category')['Total_Score'].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Sleep_Category', y='Total_Score', data=sleep_perf, ax=ax)
        
        ax.set_title("Average Total Score by Sleep Hours")
        ax.set_xlabel("Sleep Hours per Night")
        ax.set_ylabel("Average Total Score")
        
        # Add average score labels on top of bars
        for i, row in enumerate(sleep_perf.itertuples()):
            ax.text(i, row.Total_Score + 1, f"{row.Total_Score:.2f}", ha='center')
        
        st.pyplot(fig)
        
        # Stress level impact
        st.subheader("Stress Level and Performance")
        
        stress_perf = filtered_df.groupby('Stress_Level (1-10)')['Total_Score'].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(
            x='Stress_Level (1-10)',
            y='Total_Score',
            data=stress_perf,
            marker='o',
            markersize=10,
            ax=ax
        )
        
        ax.set_title("Average Total Score by Stress Level")
        ax.set_xlabel("Stress Level (1-10)")
        ax.set_ylabel("Average Total Score")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        st.pyplot(fig)
    
    # Show the raw data at the bottom
    with st.expander("View Raw Data"):
        st.dataframe(filtered_df)
else:
    st.error("Failed to load data. Please check your file path and try again.")
    
    # Provide troubleshooting guidance
    st.subheader("Troubleshooting")
    st.write("1. Make sure the CSV file is in the same directory as this script.")
    st.write("2. Check if the file path is correct.")
    st.write("3. Ensure you have proper permissions to access the file.")
    
    # File uploader as a workaround
    st.subheader("Upload the CSV file directly")
    uploaded_file = st.file_uploader("Upload the Student Dataset CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"File uploaded successfully! Found {len(df)} student records.")
        st.dataframe(df.head())