import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('winequality-red.csv')

# Data exploration
print("First 5 rows of the dataset:")
print(df.head())

print("\nNumber of missing values per column:")
print(df.isna().sum())

print("\nNumber of negative values per column:")
negative_values = (df < 0).sum()
print(negative_values)

print("\nNumber of zero values per column:")
zero_values = (df == 0).sum()
print(zero_values)

print("\nDataset shape (rows, columns):")
print(df.shape)

print("\nSummary statistics of the 'quality' column:")
print(df['quality'].describe())

# Selecting relevant columns
selected_columns = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
                       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 
                       'density', 'pH', 'sulphates', 'alcohol']]
quality_column = df[['quality']]

# Visualizing data
sns.countplot(x='quality', data=df)
plt.title('Count of Wine Quality')
plt.show()

# Correlation analysis
corr = df.corr()
idx = corr['quality'].abs().sort_values(ascending=False).index[:5]
idx_features = idx.drop('quality')
sns.heatmap(corr.loc[idx, idx], annot=True, cmap='coolwarm')
plt.title('Heatmap of Top Features Correlated with Quality')
plt.show()

# Histograms and boxplots for top correlated features
_, ax = plt.subplots(2, 2, figsize=(20, 10))
for var, axis in zip(idx_features, ax.flatten()):
    df[var].plot.hist(ax=axis, bins=20, alpha=0.7)
    axis.set_xlabel(var)
plt.tight_layout()
plt.show()

_, ax = plt.subplots(2, 2, figsize=(20, 10))
for i, var in enumerate(idx_features):
    sns.boxplot(x='quality', y=var, data=df, ax=ax.flatten()[i])
plt.tight_layout()
plt.show()

# Splitting dataset
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

print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Visualization of actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, label='Perfect Prediction')
plt.title('Wine Quality: Actual vs Predicted')
plt.xlabel('Actual Quality')
plt.ylabel('Predicted Quality')
plt.legend()
plt.show()

# Plotting learning curves
def plot_learning_curves(X, y, model):
    train_sizes, train_scores, cv_scores = learning_curve(model, X, y, cv=5, scoring='r2', n_jobs=-1)
    train_scores_mean = np.mean(train_scores, axis=1)
    cv_scores_mean = np.mean(cv_scores, axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, label='Training score', color='blue')
    plt.plot(train_sizes, cv_scores_mean, label='Cross-validation score', color='red')
    plt.xlabel('Training Size')
    plt.ylabel('R² Score')
    plt.title('Learning Curves (Wine Quality Model)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

plot_learning_curves(selected_columns, quality_column, model)
