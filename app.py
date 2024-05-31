from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
import time
import pickle

# Initialize the Flask application
app = Flask(__name__)



# Load and preprocess the dataset
def load_and_preprocess_data():
    df = pd.read_csv('dataset.csv')
    # Data Preprocessing
    df = df.drop(['id', 'mw', 'dist', 'date', 'time'], axis=1)

    # Impute missing values
    df.fillna('missing', inplace=True)

    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(df)

    # Separate features and target variable
    y = df['xm']
    X = df.drop('xm', axis=1)

    return X, y, categorical_cols


# Train the KNN model
def train_knn_model(X, y,categorical_cols):
    # Split the data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate CatBoost model using k-fold cross-validation
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    catboost_model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, cat_features=categorical_cols,
                                       silent=True)

    # Evaluate model using cross-validation
    cv_scores = cross_val_score(catboost_model, X, y, cv=kfold, scoring='neg_mean_squared_error')
    cv_r2_scores = cross_val_score(catboost_model, X, y, cv=kfold, scoring='r2')

    # Print cross-validation results
    print("Cross-Validation MSE Scores:", -cv_scores)
    print("Cross-Validation MSE Mean:", -cv_scores.mean())
    print("Cross-Validation R^2 Scores:", cv_r2_scores)
    print("Cross-Validation R^2 Mean:", cv_r2_scores.mean())

    # Fit the model on training data
    start_time = time.time()
    catboost_model.fit(X_train, y_train)
    end_time = time.time()

    # Predict on test data
    y_test_pred = catboost_model.predict(X_test)

    # Predict on training data
    y_train_pred = catboost_model.predict(X_train)

    # Calculate and print performance metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"Training MSE: {train_mse}")
    print(f"Training R^2: {train_r2}")
    print(f"Test MSE: {test_mse}")
    print(f"Test R^2: {test_r2}")
    print(f"Training time: {end_time - start_time} seconds")

    print("CatBoost Model")
    print('Mean Absolute Error:', mean_absolute_error(y_test, y_test_pred))
    print('Root Mean Squared Error:', np.sqrt(test_mse))

    return catboost_model, X_train, X_test, y_train, y_test


# Load data and train model
X, y,categorical_cols = load_and_preprocess_data()
knn_model, X_train, X_test, y_train, y_test = train_knn_model(X, y, categorical_cols)

# Save the trained KNN model
with open('knn_model.pkl', 'wb') as file:
    pickle.dump(knn_model, file)

# Load the trained model
with open('knn_model.pkl', 'rb') as file:
    knn_model = pickle.load(file)


# Home route
@app.route('/')
def home():
    return "Welcome to the Earthquake Prediction API!"


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    lat = float(request.form.get('lat'))
    long = float(request.form.get('long'))
    country = (request.form.get('country'))
    city = (request.form.get('city'))
    area = (request.form.get('area'))
    direction =   (request.form.get('direction'))
    depth = float(request.form.get('depth'))
    md = float(request.form.get('md'))
    ritcher = float(request.form.get('ritcher'))
    ms = float(request.form.get('ms'))
    mb = float(request.form.get('mb'))

    input_query = np.array([[lat, long, country, city, area, direction, depth, md, ritcher, ms, mb]])

    prediction = knn_model.predict(input_query)[0]

    # Return prediction as JSON
    return jsonify({'predicted_xm': str(prediction)})


# Main function to run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
