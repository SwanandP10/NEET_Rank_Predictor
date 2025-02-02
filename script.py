import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Step 1: Fetch Data from API Endpoints
def fetch_data(api_url):
    """Fetch data from the given API endpoint."""
    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {str(e)}")

# Step 2: Preprocess Data
def preprocess_data(current_data, historical_data):
    """Combine and preprocess current and historical quiz data."""
    try:
        # Convert JSON to DataFrames
        current_df = pd.json_normalize(current_data)
        historical_df = pd.json_normalize(historical_data)

        # Combine datasets
        combined_df = pd.concat([current_df, historical_df])

        # Normalize scores
        combined_df['normalized_score'] = (combined_df['score'] - combined_df['score'].min()) / (
                    combined_df['score'].max() - combined_df['score'].min())

        # Feature engineering: Aggregate by user
        # Replace 'response_accuracy' with 'accuracy' and parse percentage
        combined_df['accuracy'] = combined_df['accuracy'].str.replace('%', '').astype(float) / 100.0

        features = combined_df.groupby('user_id').agg({
            'normalized_score': ['mean', 'std'],
            'accuracy': 'mean'
        }).reset_index()

        # Flatten multi-level columns
        features.columns = ['user_id', 'score_mean', 'score_std', 'accuracy_mean']

        return features
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data preprocessing failed: {str(e)}")

# Step 3: Train Model
def train_model(features, neet_ranks):
    """Train a Random Forest Regressor to predict NEET rank."""
    try:
        X = features[['score_mean', 'accuracy_mean']]
        y = neet_ranks

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

# Step 4: Predict College
def predict_college(rank):
    """Predict the most likely college based on NEET rank."""
    if rank <= 100:
        return "Top Medical College (e.g., AIIMS)"
    elif rank <= 1000:
        return "Good Medical College (e.g., State Government College)"
    else:
        return "Local Medical College"

# Step 5: API Development
app = FastAPI()

class UserRequest(BaseModel):
    user_id: str

@app.post("/predict-rank")
def predict_rank(request: UserRequest):
    """API endpoint to predict NEET rank and recommend college."""
    try:
        user_id = request.user_id

        # Fetch data (replace with actual API endpoints)
        current_data = fetch_data("https://api.jsonserve.com/rJvd7g")
        historical_data = fetch_data("https://api.jsonserve.com/XgAgFJ")

        # Preprocess data
        features = preprocess_data(current_data, historical_data)

        # Simulate NEET ranks (replace with actual historical data)
        neet_ranks = np.random.randint(1, 10000, size=len(features))

        # Train model
        model = train_model(features, neet_ranks)

        # Predict rank for the given user
        user_data = features[features['user_id'] == user_id][['score_mean', 'accuracy_mean']]
        if user_data.empty:
            raise HTTPException(status_code=404, detail="User not found")

        predicted_rank = model.predict(user_data)[0]
        recommended_college = predict_college(predicted_rank)

        return {
            "user_id": user_id,
            "predicted_rank": int(predicted_rank),
            "recommended_college": recommended_college
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Step 6: Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
