import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('winequality-red.csv')
df.head()
df.isna().sum()
print("Number of negative values per column:")
negative_values = (df < 0).sum()
print(negative_values)

print("Number of zero values per column:")
zero_values = (df == 0).sum()
print(zero_values)
print("Dataset shape (rows, columns):")
print(df.shape)
df['quality'].describe()
selected_columns = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
                       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 
                       'density', 'pH', 'sulphates', 'alcohol']]

# Display the selected columns
print(selected_columns)

quality_column = df[['quality']]

# Display the selected quality column
print(quality_column)
sns.countplot(x='quality', data=df)
corr = df.corr()
idx = corr['quality'].abs().sort_values(ascending=False).index[:5]
idx_features = idx.drop('quality')
sns.heatmap(corr.loc[idx, idx])
_, ax = plt.subplots(2, 2, figsize=(20, 10))
for var, axis in zip(idx_features, ax.flatten()):
    df[var].plot.hist(ax=axis)
    axis.set_xlabel(var)
_, ax = plt.subplots(2, 2, figsize=(20, 10))
for i, var in enumerate(idx.drop('quality')):
    sns.boxplot(x='quality', y=var, data=df, ax=ax.flatten()[i])
X_train, X_test, y_train, y_test = train_test_split(selected_columns, quality_column, test_size=0.3, random_state=0)
model.fit(X_train, y_train)

# Train the model
model.fit(selected_columns, quality_column)


np.random.seed(42)
features = pd.DataFrame(np.random.rand(1500, 11), columns=[f'Feature_{i}' for i in range(1, 12)])
target = pd.DataFrame(np.random.rand(1500, 1), columns=['Target'])

# Split the data
X = selected_columns
y = quality_column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')  # Scatter plot
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, label='Perfect Prediction')  # Ideal line
plt.title('Wine Quality: Actual vs Predicted')
plt.xlabel('Actual Quality')
plt.ylabel('Predicted Quality')
plt.legend()
plt.show()

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

plot_learning_curves(X, y, model)
y_pred = model.predict(X_test)

# Accuracy: R² score
r2 = r2_score(y_test, y_pred)
print(f"R² Score (Accuracy): {r2:.4f}")

# Loss: Mean Squared Error (MSE) and Mean Absolute Error (MAE)
mse = mean_squared_error(y_test, y_pred)
mae = np.mean(np.abs(y_test - y_pred))  # Mean Absolute Error

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")