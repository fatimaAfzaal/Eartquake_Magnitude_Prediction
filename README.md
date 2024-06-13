# Eartquake_Magnitude_Prediction

This repository contains a Flask web application that uses a CatBoost regression model to predict earthquake magnitudes. The model is trained using a dataset containing various features related to earthquakes. This project demonstrates the process of loading data, preprocessing it, training a machine learning model, and deploying the model as a web service using Flask.

## Features

- **Data Preprocessing:** Load and clean the dataset, handle missing values, and prepare categorical features.
- **Model Training:** Train a CatBoost regression model using k-fold cross-validation for robust performance evaluation.
- **Model Evaluation:** Evaluate the model using metrics such as Mean Squared Error (MSE), R-squared (R²), and Mean Absolute Error (MAE).
- **Web Interface:** Provide a user-friendly web interface to input features and get earthquake magnitude predictions.
- **Model Persistence:** Save and load the trained model using pickle for easy reuse.

## File Descriptions

- `app.py`: The main Flask application file containing routes and model handling code.
- `dataset.csv`: The dataset file (not included, needs to be added by the user).
- `catboost_model.pkl`: The serialized CatBoost model file (generated after training).
- `templates/index.html`: The HTML template for the web interface.
- `requirements.txt`: The list of Python dependencies required to run the application.

## Dependencies

- Flask
- numpy
- pandas
- scikit-learn
- catboost
- pickle

## Output

![image](https://github.com/fatimaAfzaal/Eartquake_Magnitude_Prediction/assets/99525339/25cf5fbd-f705-4700-bb69-77fbc41d328f)



Feel free to contribute, provide feedback, or report issues related to this project.
