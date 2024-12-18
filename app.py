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
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.countplot(x='quality', data=df, ax=ax)
    plt.title('Count of Wine Quality')
    st.pyplot(fig)
    st.markdown("The bar graph presents a distribution of wine quality ratings. The y-axis represents the count, indicating the frequency of wines assigned to each quality score. This graph effectively highlights the distribution of wines based on their quality ratings.")
    # Correlation analysis
    st.subheader("Correlation Analysis")
    corr = df.corr()
    idx = corr['quality'].abs().sort_values(ascending=False).index[:5]
    idx_features = idx.drop('quality')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr.loc[idx, idx], annot=True, cmap='coolwarm', ax=ax)
    plt.title('Heatmap of Top Features Correlated with Quality')
    st.pyplot(fig)
    st.markdown("The heatmap provides a visual representation of the correlation between various chemical properties of wine and its quality. It uses color gradients to illustrate how these variables correlate with one another. Darker colors indicate stronger correlations, whether positive or negative, while lighter colors represent weaker correlations. This visualization helps identify which factors, such as acidity, sulfur dioxide, and alcohol content, are most strongly associated with wine quality, offering valuable insights into the relationships within the data.")
    # Histograms for top correlated features
    st.subheader("Histograms of Top Correlated Features")
    fig, ax = plt.subplots(2, 2, figsize=(20, 10))
    for var, axis in zip(idx_features, ax.flatten()):
        df[var].plot.hist(ax=axis, bins=20, alpha=0.7)
        axis.set_xlabel(var)
    st.pyplot(fig)
    st.markdown("""
The histogram provides a graphical representation of the distribution of several key chemical properties in the wine dataset: alcohol, volatile acidity, sulphates, and citric acid. Each variable is plotted on the x-axis, with the y-axis representing the frequency or count of wines that fall within specific value ranges for each property.

- **Alcohol**: The histogram shows how the alcohol content in the wines is distributed, providing insights into the most common alcohol levels in the dataset.
- **Volatile Acidity**: This variable’s histogram reveals the distribution of acetic acid in the wines, which can affect the flavor, with certain levels indicating a more sour or vinegary taste.
- **Sulphates**: The distribution of sulphate content is shown, which plays a role in preserving the wine and impacting its flavor profile.
- **Citric Acid**: The histogram for citric acid illustrates how the acidity levels vary across wines, contributing to the overall taste and balance of the wine.

By visualizing these distributions, the histogram allows for an understanding of the variation in these key chemical components, helping to identify trends, outliers, or skewed data points in the dataset.
""")
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
    st.markdown("""
The model has been trained and evaluated on the wine quality dataset. The evaluation metrics provide insight into the model’s performance:

- **Mean Squared Error (MSE)**: The MSE value is **0.3900**, which measures the average squared difference between the predicted and actual wine quality values. A lower MSE indicates better model performance, but in this case, it suggests that the model has some room for improvement in making accurate predictions.

- **R² Score**: The **R² score** is **0.4032**, which indicates that approximately 40% of the variance in the wine quality data is explained by the model. This suggests that the model provides some useful information but does not fully capture the underlying patterns in the data.

- **Actual vs Predicted**: The graph of **actual vs predicted wine quality** visually compares the true quality ratings with the model's predicted values. This allows for a better understanding of how well the model is performing, highlighting discrepancies where the predictions may differ from the actual values.

Overall, while the model shows some predictive power, there is still significant room for improvement, especially in reducing error and increasing the explained variance in the data.
""")
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
    st.markdown("If you look at the learning curve legend there is a training score and cross validation. Training data is used for training regression model then the test data is used to test performance to see the training score results. The cross validation model is used to predict the quality of validation data, however the chart indicates that both lines are far off because of the difference of model performance from data used to train unseen data. In this case both performance are bad so we can say the data is not enough.")

# Section 4: Insights
elif menu == "Insights":
    st.title("Insights")
    st.markdown("""
    Key findings from the data:
### Distribution of Wine Quality Ratings:
- The dataset contains wine quality ratings typically ranging from 0 to 10, with the majority of wines having a quality score between **5 and 6**. This suggests that most of the wines in the dataset are of average to good quality, with fewer wines rated as excellent or poor quality.

### Correlation Between Features and Quality:
- **Alcohol Content**: Alcohol content is often positively correlated with wine quality, with higher alcohol wines typically receiving higher quality ratings. This can indicate that wine with higher alcohol levels tends to be better regarded.
- **Fixed Acidity**: Wines with moderate acidity levels tend to have a better quality score, as excessive acidity or too little can negatively affect the taste.
- **pH**: pH levels, which are a measure of acidity, also have a strong influence on wine quality. Extreme values of pH (either too low or too high) can result in a poor-tasting wine.
- **Sulfur Dioxide**: Both free and total sulfur dioxide are used as preservatives in wine, but excessive amounts may negatively affect wine quality. A balance between too much and too little sulfur dioxide is likely crucial for achieving better quality wines.
- **Citric Acid**: Wines with moderate levels of citric acid might have more freshness and balance, which could contribute to higher quality ratings.
- **Residual Sugar**: Wines with high residual sugar can sometimes be perceived as sweeter, which might appeal to some consumers. However, excessive sugar could lead to lower quality ratings in dry wine categories.

### Outliers and Distribution of Features:
- **Volatile Acidity** and **Residual Sugar** might exhibit outliers. High volatile acidity can make wines taste vinegary or unpleasant, which may result in lower quality ratings. Wines with unusually high residual sugar may not align with the general preference for dry wines, affecting quality ratings.

### Impact of Winemaking Process:
- The chemical composition of wine, represented by features such as sulfur dioxide levels, residual sugar, and alcohol content, can give insights into winemaking processes. These features reflect aspects like fermentation methods, aging conditions, and grape varieties, which all influence the final quality of the wine.
""")

# Section 5: Conclusion
elif menu == "Conclusion":
    st.title("Conclusion")
    st.markdown("""
    The researchers used a Kaggle dataset on wine quality, consisting of 1,599 samples with 11 features. After training the model using this dataset, it achieved a disappointing 36% accuracy on the training data and 29% accuracy on the validation data, indicating poor predictive performance.

From the evaluation metrics and visualizations:

Mean Squared Error (MSE): The model achieved an MSE of 0.3900, suggesting that while it attempts to approximate the target wine quality ratings, its predictions deviate significantly from the actual values. This indicates room for substantial improvement.

R² Score: The R² score of 0.4032 reveals that the model explains only 40% of the variance in the data, which, while offering some predictive insight, highlights its limitations in capturing the underlying patterns.

Actual vs Predicted: The comparison between actual and predicted values shows notable discrepancies, further emphasizing the model's struggles to generalize effectively.

Learning Curves: The learning curve demonstrates a significant gap between the training and cross-validation scores. Both scores are low, suggesting that the model performs poorly on both the training and validation data. This may indicate insufficient data, suboptimal model selection, or inadequate feature engineering.

Overall, the model provides limited predictive power and fails to generalize well to unseen data. Future efforts should focus on improving data preprocessing, exploring alternative models, and possibly collecting or synthesizing additional data to enhance performance.
    """)








