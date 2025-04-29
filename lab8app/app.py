from fastapi import FastAPI, HTTPException
import uvicorn
import joblib
from pydantic import BaseModel
import pandas as pd
# import streamlit as st


# st.title("Credit Judge: Will you repay a loan?")
# st.markdown("### Fill in the form below and hit JUDGE!")

app = FastAPI(
    title="Credit Judge",
    description="Classify loan seekers as either Safe or Risky.",
    version="0.1",
)

@app.get('/')
def main():
	return {'message': 'This is a model for classifying loan risk'}

class LoanPredictionRequest(BaseModel):
    current_loan_amount: float
    term: str  # 'Short Term' or 'Long Term'
    credit_score: float
    annual_income: float
    home_ownership: str  # 'Home Mortgage', 'Rent', etc.
    monthly_debt: float
    years_of_credit_history: float
    current_credit_balance: float
    maximum_open_credit: float
    bankruptcies: int
    

@app.on_event('startup')
def load_artifacts():
    global model_pipeline
    global label_encoders
    model_pipeline = joblib.load("../src/bank_model.pkl")
    label_encoders = joblib.load("../src/feature_encoders.pkl")

# Full feature list based on the model's training dataset
all_columns = [
    'Current Loan Amount', 'Term', 'Credit Score', 'Annual Income', 'Years in current job', 
    'Home Ownership', 'Purpose', 'Monthly Debt', 'Years of Credit History', 
    'Months since last delinquent', 'Number of Open Accounts', 'Number of Credit Problems', 
    'Current Credit Balance', 'Maximum Open Credit', 'Bankruptcies', 'Tax Liens'
]


# Defining path operation for /predict endpoint
@app.post('/predict')
def predict(data : LoanPredictionRequest):
    input_data = data.dict()

    try:
        # Encode categorical fields like 'home_ownership' and 'term'
        input_data['home_ownership'] = label_encoders['Home Ownership'].transform([input_data['home_ownership']])[0]
        input_data['term'] = label_encoders['Term'].transform([input_data['term']])[0]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Encoding error: {e}")

    # Fill in missing columns with default values, as done in your original code
    for col in all_columns:
        if col not in input_data:
            if col in ["Term", "Home Ownership", "Purpose"]:  # Categorical columns
                input_data[col] = label_encoders[col].classes_[0]  # Default to the first class
            else:  # Numeric columns
                input_data[col] = 0.0

    # Convert the input data to a DataFrame
    input_df = pd.DataFrame([input_data])[all_columns]

    # Apply label encoding to categorical columns
    label_cols = ['Term', 'Home Ownership', 'Purpose']
    for col in label_cols:
        if col in input_data:
            le = label_encoders[col]
            input_df[col] = le.transform(input_df[col])

    # Make the prediction using your pre-trained model
    prediction = model_pipeline.predict(input_df)

    # Return the result (1 = Safe, 0 = Risky)
    loan_status = 'Risky' if prediction[0] == 0 else 'Safe'
    
    # Return the result
    return {'Loan Status': loan_status}
    

