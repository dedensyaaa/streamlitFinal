import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Set the page configuration
st.set_page_config(page_title="Wine Quality Prediction", layout="wide")

# Load dataset
df = pd.read_csv('winequality-red.csv')



# Sidebar menu for navigation
st.sidebar.title("Presentation Menu")
menu = st.sidebar.radio(
    "Select a Section:",
    ["Overview", "Data Visualization", "Insights", "Conclusion"]
)

if menu == "Overview":
    st.title("Wine Quality Prediction")
    st.write("### First 5 rows of the dataset:")
    st.write(df.head())
    st.write("### Number of missing values per column:")
    st.write(df.isna().sum())
    st.write("### Number of negative values per column:")
    negative_values = (df < 0).sum()
    st.write(negative_values)
    st.write("### Number of zero values per column:")
    zero_values = (df == 0).sum()
    st.write(zero_values)
    st.write("### Dataset shape (rows, columns):")
    st.write(df.shape)
    st.write("### Summary statistics of the 'quality' column:")
    st.write(df['quality'].describe())
    
elif menu == "Data Visualization":
    st.header("Data Visualization")
    st.subheader("Count of Wine Quality")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(x='quality', data=df, ax=ax)
    plt.title('Count of Wine Quality')
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("""
    <h3 style='font-size: 18px;'>The bar graph presents a distribution of wine quality ratings. The y-axis represents the count, indicating the frequency of wines assigned to each quality score. This graph effectively highlights the distribution of wines based on their quality ratings.</h3>
""")
    # Correlation analysis
    st.subheader("Correlation Analysis")
    corr = df.corr()
    idx = corr['quality'].abs().sort_values(ascending=False).index[:5]
    idx_features = idx.drop('quality')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr.loc[idx, idx], annot=True, cmap='coolwarm', ax=ax)
    plt.title('Heatmap of Top Features Correlated with Quality')
    st.pyplot(fig)
    # Histograms for top correlated features
    st.subheader("Histograms of Top Correlated Features")
    fig, ax = plt.subplots(2, 2, figsize=(20, 10))
    for var, axis in zip(idx_features, ax.flatten()):
        df[var].plot.hist(ax=axis, bins=20, alpha=0.7)
        axis.set_xlabel(var)
    st.pyplot(fig)
    # Splitting dataset
    st.header("Model Training and Evaluation")
    selected_columns = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
                           'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 
                           'density', 'pH', 'sulphates', 'alcohol']]
    quality_column = df[['quality']]
    X_train, X_test, y_train, y_test = train_test_split(selected_columns, quality_column, test_size=0.2, random_state=42)
    # Scaling features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Predictions
    y_pred = model.predict(X_test)
    # Evaluating the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"### Mean Squared Error: {mse:.4f}")
    st.write(f"### R² Score: {r2:.4f}")
    # Visualization of actual vs predicted
    st.subheader("Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, label='Perfect Prediction')
    plt.title('Wine Quality: Actual vs Predicted')
    plt.xlabel('Actual Quality')
    plt.ylabel('Predicted Quality')
    plt.legend()
    st.pyplot(fig)
    # Learning curves
    st.subheader("Learning Curves")
    def plot_learning_curves(X, y, model):
        train_sizes, train_scores, cv_scores = learning_curve(model, X, y, cv=5, scoring='r2', n_jobs=-1)
        train_scores_mean = np.mean(train_scores, axis=1)
        cv_scores_mean = np.mean(cv_scores, axis=1)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_sizes, train_scores_mean, label='Training score', color='blue')
        ax.plot(train_sizes, cv_scores_mean, label='Cross-validation score', color='red')
        plt.xlabel('Training Size')
        plt.ylabel('R² Score')
        plt.title('Learning Curves (Wine Quality Model)')
        plt.legend(loc='best')
        plt.grid(True)
        return fig
    fig = plot_learning_curves(selected_columns, quality_column, model)
    st.pyplot(fig)

# Section 4: Insights
elif menu == "Insights":
    st.title("Insights")
    st.markdown("""
    Key findings from the data:
    - **Top Performer**: Charlie with a score of 95.
    - **Age Group with Highest Score**: 35 years.
    - **Average Score**: 90.
    """)

# Section 5: Conclusion
elif menu == "Conclusion":
    st.title("Conclusion")
    st.markdown("""
    - **Summary**: The data reveals trends in performance by age group.
    - **Next Steps**: Consider additional analysis for deeper insights.
    Thank you for using this app!
    """)








