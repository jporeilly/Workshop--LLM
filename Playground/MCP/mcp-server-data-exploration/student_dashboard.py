import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Student Performance Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #333;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .card {
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
    }
    .metric-container {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        min-width: 180px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data()
def load_data():
    # Replace this with the actual path to your CSV file
    df = pd.read_csv("Students_Grading_Dataset.csv")
    
    # Create additional feature columns for analysis
    df['Sleep_Category'] = pd.cut(
        df['Sleep_Hours_per_Night'], 
        bins=[3.9, 5.0, 6.0, 7.0, 8.0, 9.1], 
        labels=['4-5', '5-6', '6-7', '7-8', '8-9']
    )
    
    df['Study_Hours_Category'] = pd.cut(
        df['Study_Hours_per_Week'],
        bins=[4.9, 10.0, 15.0, 20.0, 25.0, 30.1],
        labels=['5-10', '10-15', '15-20', '20-25', '25-30']
    )
    
    df['Stress_Category'] = pd.cut(
        df['Stress_Level (1-10)'],
        bins=[0, 3, 6, 10],
        labels=['Low (1-3)', 'Medium (4-6)', 'High (7-10)']
    )
    
    # Calculate if a student passed or failed
    df['Pass_Status'] = df['Grade'].apply(lambda x: 'Failed' if x == 'F' else 'Passed')
    
    return df

# Load the data
try:
    df = load_data()
    # Show a success message
    st.success("Data loaded successfully!")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar filters
st.sidebar.title("Filters")

# Department filter
departments = ['All'] + sorted(df['Department'].unique().tolist())
selected_department = st.sidebar.selectbox("Select Department", departments)

# Gender filter
genders = ['All'] + sorted(df['Gender'].unique().tolist())
selected_gender = st.sidebar.selectbox("Select Gender", genders)

# Grade filter
grades = ['All'] + sorted(df['Grade'].unique().tolist())
selected_grade = st.sidebar.selectbox("Select Grade", grades)

# Additional filters in an expander
with st.sidebar.expander("More Filters"):
    # Family income level filter
    income_levels = ['All'] + sorted(df['Family_Income_Level'].unique().tolist())
    selected_income = st.selectbox("Family Income Level", income_levels)
    
    # Internet access filter
    internet_options = ['All'] + sorted(df['Internet_Access_at_Home'].unique().tolist())
    selected_internet = st.selectbox("Internet Access", internet_options)
    
    # Extracurricular activities filter
    extra_options = ['All'] + sorted(df['Extracurricular_Activities'].unique().tolist())
    selected_extra = st.selectbox("Extracurricular Activities", extra_options)
    
    # Stress level range
    stress_min, stress_max = st.slider(
        "Stress Level Range",
        min_value=1,
        max_value=10,
        value=(1, 10)
    )

# Apply filters to dataframe
filtered_df = df.copy()

if selected_department != 'All':
    filtered_df = filtered_df[filtered_df['Department'] == selected_department]
    
if selected_gender != 'All':
    filtered_df = filtered_df[filtered_df['Gender'] == selected_gender]
    
if selected_grade != 'All':
    filtered_df = filtered_df[filtered_df['Grade'] == selected_grade]
    
if selected_income != 'All':
    filtered_df = filtered_df[filtered_df['Family_Income_Level'] == selected_income]
    
if selected_internet != 'All':
    filtered_df = filtered_df[filtered_df['Internet_Access_at_Home'] == selected_internet]
    
if selected_extra != 'All':
    filtered_df = filtered_df[filtered_df['Extracurricular_Activities'] == selected_extra]

filtered_df = filtered_df[(filtered_df['Stress_Level (1-10)'] >= stress_min) & 
                         (filtered_df['Stress_Level (1-10)'] <= stress_max)]

# About section in the sidebar
with st.sidebar.expander("About Dashboard"):
    st.write("""
    This dashboard visualizes the student performance dataset with 5,000 student records.
    
    The data includes academic metrics, demographics, and lifestyle factors to help analyze
    patterns in student performance.
    
    Use the filters to explore different segments of the student population.
    """)
    
    st.write("**Dataset Statistics:**")
    st.write(f"- Total Students: {len(df)}")
    st.write(f"- Departments: {', '.join(df['Department'].unique())}")
    st.write(f"- Grade Range: {', '.join(sorted(df['Grade'].unique()))}")

# Main dashboard content
st.markdown('<div class="main-header">Student Performance Analysis Dashboard</div>', unsafe_allow_html=True)

# Display filtered data count
st.write(f"Showing data for **{len(filtered_df)}** students out of {len(df)} total students")

# Key metrics
st.markdown('<div class="sub-header">Key Performance Metrics</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_total = filtered_df['Total_Score'].mean()
    st.metric("Average Total Score", f"{avg_total:.2f}")

with col2:
    passing_rate = (filtered_df['Pass_Status'] == 'Passed').mean() * 100
    st.metric("Passing Rate", f"{passing_rate:.1f}%")

with col3:
    avg_attendance = filtered_df['Attendance (%)'].mean()
    st.metric("Average Attendance", f"{avg_attendance:.1f}%")

with col4:
    avg_study = filtered_df['Study_Hours_per_Week'].mean()
    st.metric("Avg Study Hours/Week", f"{avg_study:.1f}")

# Create tabs for different analysis views
tab1, tab2, tab3, tab4 = st.tabs(["Grade Analysis", "Academic Metrics", "Lifestyle Factors", "Demographic Impact"])

with tab1:
    st.markdown('<div class="sub-header">Grade Distribution and Analysis</div>', unsafe_allow_html=True)
    
    # Grade distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Grade Distribution")
        grade_counts = filtered_df['Grade'].value_counts().reset_index()
        grade_counts.columns = ['Grade', 'Count']
        
        # Sort grades in the correct order
        correct_order = ['A', 'B', 'C', 'D', 'F']
        grade_counts['Grade'] = pd.Categorical(grade_counts['Grade'], categories=correct_order, ordered=True)
        grade_counts = grade_counts.sort_values('Grade')
        
        fig = px.bar(
            grade_counts, 
            x='Grade', 
            y='Count',
            color='Grade',
            color_discrete_map={
                'A': '#4CAF50',  # Green
                'B': '#8BC34A',  # Light green
                'C': '#FFC107',  # Yellow
                'D': '#FF9800',  # Orange
                'F': '#F44336',  # Red
            },
            text='Count'
        )
        
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Department Grade Distribution")
        
        dept_grade = pd.crosstab(
            filtered_df['Department'], 
            filtered_df['Grade'],
            normalize='index'
        ) * 100
        
        # Reorder columns to match grade order
        if not dept_grade.empty and set(correct_order).issubset(set(dept_grade.columns)):
            dept_grade = dept_grade[correct_order]
        
        fig = px.bar(
            dept_grade.reset_index().melt(id_vars=['Department'], var_name='Grade', value_name='Percentage'),
            x='Department',
            y='Percentage',
            color='Grade',
            barmode='group',
            color_discrete_map={
                'A': '#4CAF50',
                'B': '#8BC34A',
                'C': '#FFC107',
                'D': '#FF9800',
                'F': '#F44336',
            },
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics by grade
    st.subheader("Performance Metrics by Grade")
    
    grade_performance = filtered_df.groupby('Grade')[
        ['Attendance (%)', 'Midterm_Score', 'Final_Score', 'Assignments_Avg', 
         'Quizzes_Avg', 'Participation_Score', 'Projects_Score', 'Total_Score']
    ].mean().reset_index()
    
    # Sort by grade order
    grade_performance['Grade'] = pd.Categorical(grade_performance['Grade'], categories=correct_order, ordered=True)
    grade_performance = grade_performance.sort_values('Grade')
    
    selected_metrics = st.multiselect(
        "Select Performance Metrics to Compare",
        options=['Attendance (%)', 'Midterm_Score', 'Final_Score', 'Assignments_Avg', 
                'Quizzes_Avg', 'Participation_Score', 'Projects_Score', 'Total_Score'],
        default=['Midterm_Score', 'Final_Score', 'Total_Score']
    )
    
    if selected_metrics:
        fig = px.line(
            grade_performance,
            x='Grade',
            y=selected_metrics,
            markers=True,
            line_shape='linear',
            height=500
        )
        
        fig.update_layout(
            xaxis_title='Grade',
            yaxis_title='Score',
            legend_title='Metric'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select at least one performance metric to display.")

with tab2:
    st.markdown('<div class="sub-header">Academic Performance Metrics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Score Distribution")
        
        score_vars = st.selectbox(
            "Select Score Variable",
            options=['Total_Score', 'Midterm_Score', 'Final_Score', 'Assignments_Avg', 
                    'Quizzes_Avg', 'Participation_Score', 'Projects_Score'],
            index=0
        )
        
        fig = px.histogram(
            filtered_df,
            x=score_vars,
            nbins=30,
            marginal='box',
            color_discrete_sequence=['#1E88E5'],
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Score Correlation")
        
        corr_x = st.selectbox(
            "Select X-axis Score",
            options=['Midterm_Score', 'Final_Score', 'Assignments_Avg', 
                    'Quizzes_Avg', 'Participation_Score', 'Projects_Score'],
            index=0
        )
        
        corr_y = st.selectbox(
            "Select Y-axis Score",
            options=['Final_Score', 'Total_Score', 'Midterm_Score', 'Assignments_Avg', 
                    'Quizzes_Avg', 'Participation_Score', 'Projects_Score'],
            index=0
        )
        
        if corr_x != corr_y:
            fig = px.scatter(
                filtered_df,
                x=corr_x,
                y=corr_y,
                color='Grade',
                trendline='ols',
                color_discrete_map={
                    'A': '#4CAF50',
                    'B': '#8BC34A',
                    'C': '#FFC107',
                    'D': '#FF9800',
                    'F': '#F44336',
                },
                opacity=0.7
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate correlation
            correlation = filtered_df[[corr_x, corr_y]].corr().iloc[0,1]
            st.write(f"Correlation between {corr_x} and {corr_y}: **{correlation:.3f}**")
        else:
            st.warning("Please select different variables for X and Y axes.")
    
    # Heatmap of correlations
    st.subheader("Correlation Heatmap of Academic Metrics")
    
    score_columns = ['Attendance (%)', 'Midterm_Score', 'Final_Score', 'Assignments_Avg', 
                     'Quizzes_Avg', 'Participation_Score', 'Projects_Score', 'Total_Score',
                     'Study_Hours_per_Week', 'Stress_Level (1-10)', 'Sleep_Hours_per_Night']
    
    correlation_matrix = filtered_df[score_columns].corr().round(2)
    
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        aspect="auto",
        height=650
    )
    
    fig.update_layout(
        title="Correlation Between Academic and Lifestyle Metrics"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown('<div class="sub-header">Lifestyle Factors and Academic Performance</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Study Hours Impact")
        
        study_perf = filtered_df.groupby('Study_Hours_Category')['Total_Score'].mean().reset_index()
        
        # Ensure categories are in the right order
        category_order = ['5-10', '10-15', '15-20', '20-25', '25-30']
        study_perf['Study_Hours_Category'] = pd.Categorical(
            study_perf['Study_Hours_Category'], 
            categories=category_order, 
            ordered=True
        )
        study_perf = study_perf.sort_values('Study_Hours_Category')
        
        fig = px.bar(
            study_perf,
            x='Study_Hours_Category',
            y='Total_Score',
            color='Total_Score',
            color_continuous_scale='Viridis',
            text_auto='.2f'
        )
        
        fig.update_traces(textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Sleep Hours Impact")
        
        sleep_perf = filtered_df.groupby('Sleep_Category')['Total_Score'].mean().reset_index()
        
        # Ensure categories are in the right order
        sleep_order = ['4-5', '5-6', '6-7', '7-8', '8-9']
        sleep_perf['Sleep_Category'] = pd.Categorical(
            sleep_perf['Sleep_Category'], 
            categories=sleep_order, 
            ordered=True
        )
        sleep_perf = sleep_perf.sort_values('Sleep_Category')
        
        fig = px.bar(
            sleep_perf,
            x='Sleep_Category',
            y='Total_Score',
            color='Total_Score',
            color_continuous_scale='Viridis',
            text_auto='.2f'
        )
        
        fig.update_traces(textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Stress level and performance
    st.subheader("Stress Level and Academic Performance")
    
    # Create dataframe with stress levels and scores
    stress_scores = filtered_df.groupby('Stress_Level (1-10)')[
        ['Total_Score', 'Midterm_Score', 'Final_Score']
    ].mean().reset_index()
    
    fig = px.line(
        stress_scores,
        x='Stress_Level (1-10)',
        y=['Total_Score', 'Midterm_Score', 'Final_Score'],
        markers=True,
        line_shape='linear',
        height=500
    )
    
    fig.update_layout(
        xaxis_title='Stress Level (1-10)',
        yaxis_title='Score',
        legend_title='Metric'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Extracurricular activities effect
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Extracurricular Activities Effect")
        
        extra_perf = filtered_df.groupby('Extracurricular_Activities')[
            ['Total_Score', 'Study_Hours_per_Week']
        ].mean().reset_index()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(
                x=extra_perf['Extracurricular_Activities'],
                y=extra_perf['Total_Score'],
                name='Total Score',
                marker_color='#1E88E5'
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=extra_perf['Extracurricular_Activities'],
                y=extra_perf['Study_Hours_per_Week'],
                name='Study Hours',
                marker_color='#FFC107',
                mode='markers+lines',
                marker=dict(size=12)
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title_text="Total Score and Study Hours by Extracurricular Activities",
            height=400
        )
        
        fig.update_xaxes(title_text="Extracurricular Activities")
        fig.update_yaxes(title_text="Total Score", secondary_y=False)
        fig.update_yaxes(title_text="Study Hours per Week", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Internet Access Effect")
        
        internet_perf = filtered_df.groupby('Internet_Access_at_Home')[
            ['Total_Score', 'Assignments_Avg']
        ].mean().reset_index()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(
                x=internet_perf['Internet_Access_at_Home'],
                y=internet_perf['Total_Score'],
                name='Total Score',
                marker_color='#1E88E5'
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=internet_perf['Internet_Access_at_Home'],
                y=internet_perf['Assignments_Avg'],
                name='Assignments Average',
                marker_color='#4CAF50',
                mode='markers+lines',
                marker=dict(size=12)
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title_text="Total Score and Assignments by Internet Access",
            height=400
        )
        
        fig.update_xaxes(title_text="Internet Access at Home")
        fig.update_yaxes(title_text="Total Score", secondary_y=False)
        fig.update_yaxes(title_text="Assignments Average", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown('<div class="sub-header">Demographic and Socioeconomic Factors</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gender Performance Comparison")
        
        gender_perf = filtered_df.groupby('Gender')[
            ['Total_Score', 'Midterm_Score', 'Final_Score']
        ].mean().reset_index()
        
        fig = px.bar(
            gender_perf,
            x='Gender',
            y=['Total_Score', 'Midterm_Score', 'Final_Score'],
            barmode='group',
            height=400,
            color_discrete_sequence=['#1E88E5', '#4CAF50', '#FFC107']
        )
        
        fig.update_layout(
            xaxis_title='Gender',
            yaxis_title='Score',
            legend_title='Metric'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Gender grade distribution
        gender_grade = pd.crosstab(
            filtered_df['Gender'], 
            filtered_df['Grade'],
            normalize='index'
        ) * 100
        
        # Reorder columns to match grade order
        if not gender_grade.empty and set(correct_order).issubset(set(gender_grade.columns)):
            gender_grade = gender_grade[correct_order]
        
        fig = px.bar(
            gender_grade.reset_index().melt(id_vars=['Gender'], var_name='Grade', value_name='Percentage'),
            x='Gender',
            y='Percentage',
            color='Grade',
            barmode='stack',
            color_discrete_map={
                'A': '#4CAF50',
                'B': '#8BC34A',
                'C': '#FFC107',
                'D': '#FF9800',
                'F': '#F44336',
            },
            height=400
        )
        
        fig.update_layout(
            xaxis_title='Gender',
            yaxis_title='Percentage',
            legend_title='Grade'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Age Impact on Performance")
        
        age_perf = filtered_df.groupby('Age')[
            ['Total_Score', 'Study_Hours_per_Week', 'Stress_Level (1-10)']
        ].mean().reset_index()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(
                x=age_perf['Age'],
                y=age_perf['Total_Score'],
                name='Total Score',
                marker_color='#1E88E5'
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=age_perf['Age'],
                y=age_perf['Study_Hours_per_Week'],
                name='Study Hours',
                marker_color='#4CAF50',
                mode='markers+lines',
                marker=dict(size=10)
            ),
            secondary_y=True
        )
        
        fig.add_trace(
            go.Scatter(
                x=age_perf['Age'],
                y=age_perf['Stress_Level (1-10)'],
                name='Stress Level',
                marker_color='#F44336',
                mode='markers+lines',
                marker=dict(size=10)
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title_text="Performance Metrics by Age",
            height=400
        )
        
        fig.update_xaxes(title_text="Age")
        fig.update_yaxes(title_text="Total Score", secondary_y=False)
        fig.update_yaxes(title_text="Hours / Level", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Socioeconomic factors
    st.subheader("Socioeconomic Factors Impact")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Family Income Level Effect")
        
        income_perf = filtered_df.groupby('Family_Income_Level')['Total_Score'].mean().reset_index()
        
        # Set correct order for income levels
        income_order = ['Low', 'Medium', 'High']
        income_perf['Family_Income_Level'] = pd.Categorical(
            income_perf['Family_Income_Level'], 
            categories=income_order, 
            ordered=True
        )
        income_perf = income_perf.sort_values('Family_Income_Level')
        
        fig = px.bar(
            income_perf,
            x='Family_Income_Level',
            y='Total_Score',
            color='Family_Income_Level',
            color_discrete_sequence=['#FFC107', '#1E88E5', '#4CAF50'],
            text_auto='.2f',
            height=400
        )
        
        fig.update_traces(textposition='outside')
        
        fig.update_layout(
            xaxis_title='Family Income Level',
            yaxis_title='Total Score',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Family income and stress
        income_stress = filtered_df.groupby('Family_Income_Level')['Stress_Level (1-10)'].mean().reset_index()
        
        # Reorder categories
        income_stress['Family_Income_Level'] = pd.Categorical(
            income_stress['Family_Income_Level'], 
            categories=income_order, 
            ordered=True
        )
        income_stress = income_stress.sort_values('Family_Income_Level')
        
        fig = px.bar(
            income_stress,
            x='Family_Income_Level',
            y='Stress_Level (1-10)',
            color='Family_Income_Level',
            color_discrete_sequence=['#FFC107', '#1E88E5', '#4CAF50'],
            text_auto='.2f',
            height=400
        )
        
        fig.update_traces(textposition='outside')
        
        fig.update_layout(
            xaxis_title='Family Income Level',
            yaxis_title='Stress Level (1-10)',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Parent education effect - only if we have data
        if 'Parent_Education_Level' in filtered_df.columns and not filtered_df['Parent_Education_Level'].isna().all():
            st.subheader("Parent Education Level Effect")
            
            # Filter out NaN values
            parent_edu_df = filtered_df.dropna(subset=['Parent_Education_Level'])
            
            if not parent_edu_df.empty:
                parent_edu_perf = parent_edu_df.groupby('Parent_Education_Level')['Total_Score'].mean().reset_index()
                
                # Set correct order for education levels
                edu_order = ['None', 'High School', 'Bachelor\'s', 'Master\'s', 'PhD']
                available_edu = [edu for edu in edu_order if edu in parent_edu_perf['Parent_Education_Level'].unique()]
                
                if available_edu:
                    parent_edu_perf['Parent_Education_Level'] = pd.Categorical(
                        parent_edu_perf['Parent_Education_Level'], 
                        categories=available_edu, 
                        ordered=True
                    )
                    parent_edu_perf = parent_edu_perf.sort_values('Parent_Education_Level')
                
                fig = px.bar(
                    parent_edu_perf,
                    x='Parent_Education_Level',
                    y='Total_Score',
                    color='Parent_Education_Level',
                    text_auto='.2f',
                    height=400
                )
                
                fig.update_traces(textposition='outside')
                
                fig.update_layout(
                    xaxis_title='Parent Education Level',
                    yaxis_title='Total Score',
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Parent education and grade distribution
                parent_edu_grade = pd.crosstab(
                    parent_edu_df['Parent_Education_Level'], 
                    parent_edu_df['Grade'],
                    normalize='index'
                ) * 100
                
                # Reorder columns to match grade order
                if not parent_edu_grade.empty and set(correct_order).issubset(set(parent_edu_grade.columns)):
                    parent_edu_grade = parent_edu_grade[correct_order]
                
                fig = px.bar(
                    parent_edu_grade.reset_index().melt(
                        id_vars=['Parent_Education_Level'], 
                        var_name='Grade', 
                        value_name='Percentage'
                    ),
                    x='Parent_Education_Level',
                    y='Percentage',
                    color='Grade',
                    barmode='stack',
                    color_discrete_map={
                        'A': '#4CAF50',
                        'B': '#8BC34A',
                        'C': '#FFC107',
                        'D': '#FF9800',
                        'F': '#F44336',
                    },
                    height=400
                )
                
                fig.update_layout(
                    xaxis_title='Parent Education Level',
                    yaxis_title='Percentage',
                    legend_title='Grade'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for Parent Education Level analysis with current filters.")
        else:
            st.info("Parent Education Level data is not available.")

    # Advanced analysis section
    st.markdown('<div class="sub-header">Advanced Analysis</div>', unsafe_allow_html=True)
    
    with st.expander("Top and Bottom Performers Analysis"):
        # Number of students to analyze
        n_students = st.slider("Number of students to analyze", 50, 500, 250, 50)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Top {n_students} Students Profile")
            
            top_students = filtered_df.nlargest(n_students, 'Total_Score')
            top_stats = top_students[['Study_Hours_per_Week', 'Stress_Level (1-10)', 'Sleep_Hours_per_Night', 'Attendance (%)']].mean()
            
            top_metrics = {
                "Avg. Study Hours": f"{top_stats['Study_Hours_per_Week']:.2f}",
                "Avg. Stress Level": f"{top_stats['Stress_Level (1-10)']:.2f}",
                "Avg. Sleep Hours": f"{top_stats['Sleep_Hours_per_Night']:.2f}",
                "Avg. Attendance": f"{top_stats['Attendance (%)']:.2f}%"
            }
            
            # Grade distribution for top performers
            top_grades = top_students['Grade'].value_counts(normalize=True) * 100
            
            fig = px.pie(
                values=top_grades.values,
                names=top_grades.index,
                title=f"Grade Distribution - Top {n_students} Students",
                color=top_grades.index,
                color_discrete_map={
                    'A': '#4CAF50',
                    'B': '#8BC34A',
                    'C': '#FFC107',
                    'D': '#FF9800',
                    'F': '#F44336',
                },
                hole=0.4
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=300)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            for label, value in top_metrics.items():
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.subheader(f"Bottom {n_students} Students Profile")
            
            bottom_students = filtered_df.nsmallest(n_students, 'Total_Score')
            bottom_stats = bottom_students[['Study_Hours_per_Week', 'Stress_Level (1-10)', 'Sleep_Hours_per_Night', 'Attendance (%)']].mean()
            
            bottom_metrics = {
                "Avg. Study Hours": f"{bottom_stats['Study_Hours_per_Week']:.2f}",
                "Avg. Stress Level": f"{bottom_stats['Stress_Level (1-10)']:.2f}",
                "Avg. Sleep Hours": f"{bottom_stats['Sleep_Hours_per_Night']:.2f}",
                "Avg. Attendance": f"{bottom_stats['Attendance (%)']:.2f}%"
            }
            
            # Grade distribution for bottom performers
            bottom_grades = bottom_students['Grade'].value_counts(normalize=True) * 100
            
            fig = px.pie(
                values=bottom_grades.values,
                names=bottom_grades.index,
                title=f"Grade Distribution - Bottom {n_students} Students",
                color=bottom_grades.index,
                color_discrete_map={
                    'A': '#4CAF50',
                    'B': '#8BC34A',
                    'C': '#FFC107',
                    'D': '#FF9800',
                    'F': '#F44336',
                },
                hole=0.4
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=300)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            for label, value in bottom_metrics.items():
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Compare the differences
        diff_metrics = {key: float(top_metrics[key].replace('%', '')) - float(bottom_metrics[key].replace('%', '')) for key in top_metrics}
        
        st.subheader("Differences Between Top and Bottom Performers")
        
        diff_df = pd.DataFrame({
            'Metric': list(diff_metrics.keys()),
            'Difference': list(diff_metrics.values())
        })
        
        fig = px.bar(
            diff_df,
            x='Metric',
            y='Difference',
            color='Difference',
            color_continuous_scale='RdBu',
            text_auto='.2f'
        )
        
        fig.update_traces(textposition='outside')
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # 3D visualization of multiple factors
    with st.expander("3D Visualization"):
        st.subheader("3D Visualization of Multiple Factors")
        
        x_var = st.selectbox(
            "Select X-axis Variable",
            options=[
                'Study_Hours_per_Week', 'Sleep_Hours_per_Night', 'Stress_Level (1-10)', 
                'Attendance (%)', 'Midterm_Score', 'Final_Score'
            ],
            index=0
        )
        
        y_var = st.selectbox(
            "Select Y-axis Variable",
            options=[
                'Sleep_Hours_per_Night', 'Study_Hours_per_Week', 'Stress_Level (1-10)', 
                'Attendance (%)', 'Midterm_Score', 'Final_Score'
            ],
            index=0
        )
        
        z_var = st.selectbox(
            "Select Z-axis Variable",
            options=['Total_Score'],
            index=0
        )
        
        color_var = st.selectbox(
            "Color by",
            options=['Grade', 'Department', 'Gender', 'Extracurricular_Activities', 'Family_Income_Level'],
            index=0
        )
        
        if x_var != y_var:
            # Take a sample if dataset is large to avoid overcrowding
            plot_df = filtered_df
            if len(filtered_df) > 1000:
                plot_df = filtered_df.sample(1000, random_state=42)
            
            fig = px.scatter_3d(
                plot_df,
                x=x_var,
                y=y_var,
                z=z_var,
                color=color_var,
                opacity=0.7,
                size_max=10,
                height=700
            )
            
            fig.update_layout(
                scene=dict(
                    xaxis_title=x_var,
                    yaxis_title=y_var,
                    zaxis_title=z_var
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select different variables for X and Y axes.")

# Allow the user to view and download the filtered data
with st.expander("View Filtered Data"):
    st.write(filtered_df)
    
    # Create a download button for the filtered data
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_student_data.csv",
        mime="text/csv"
    )

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #ddd; color: #666;">
    <p>Student Performance Analysis Dashboard</p>
    <p>Created with Streamlit and Python</p>
</div>
""", unsafe_allow_html=True)