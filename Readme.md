# NEET Rank Prediction API

This project provides an API for predicting NEET (National Eligibility cum Entrance Test) ranks and recommending medical colleges based on quiz data. It uses machine learning (Random Forest Regressor) to predict NEET ranks based on users' performance in a set of quizzes.

## Features
- Fetches current and historical quiz data from external APIs.
- Processes and normalizes quiz data to create useful features.
- Trains a Random Forest model to predict NEET ranks.
- Recommends a college based on the predicted NEET rank.
- API endpoint for predicting the NEET rank and recommending colleges.

## Technologies Used
- **FastAPI**: For creating the web API.
- **Pydantic**: For data validation.
- **Requests**: For fetching data from external APIs.
- **Pandas**: For data manipulation and processing.
- **NumPy**: For generating simulated NEET ranks and handling numerical data.
- **Scikit-learn**: For training the Random Forest Regressor model.
- **Uvicorn**: For serving the FastAPI application.

## Setup

pip install -r requirements.txt
python3 script.py
go to 27.0.0.1:8000/docs in your browser