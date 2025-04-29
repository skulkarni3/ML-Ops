import requests

user_data = {
  "current_loan_amount": 500,
  "term": "Short Term",
  "credit_score": 450,
  "annual_income": 40000,
  "home_ownership": "Rent",
  "monthly_debt": 20,
  "years_of_credit_history": 3,
  "current_credit_balance": 600,
  "maximum_open_credit": 700,
  "bankruptcies": 0
}

url = 'http://127.0.0.1:8000/predict'
response = requests.post(url, json=user_data)
print(response.json())

# import joblib
# import pandas as pd

# # Load the model
# model = joblib.load("../src/bank_model.pkl")
# label_encoders = joblib.load("../src/feature_encoders.pkl")
# print(model)

# # Test with a sample input (the same format as your API input)
# sample_input = {
#     "Current Loan Amount": 445412.0,
#     "Term": "Short Term",
#     "Credit Score": 709.0,
#     "Annual Income": 1167493.0,
#     "Home Ownership": "Home Mortgage",
#     "Monthly Debt": 5214.74,
#     "Years of Credit History": 17.2,
#     "Current Credit Balance": 228190.0,
#     "Maximum Open Credit": 416746.0,
#     "Bankruptcies": 1
# }
# # Full feature list based on the model's training dataset (include all features used during training)
# all_columns =['Current Loan Amount', 'Term', 'Credit Score',
#        'Annual Income', 'Years in current job', 'Home Ownership', 'Purpose',
#        'Monthly Debt', 'Years of Credit History',
#        'Months since last delinquent', 'Number of Open Accounts',
#        'Number of Credit Problems', 'Current Credit Balance',
#        'Maximum Open Credit', 'Bankruptcies', 'Tax Liens']

# # Ensure that every column in `all_columns` is present in `sample_input` and add missing columns with default values
# for col in all_columns:
#     if col not in sample_input:
#         if col in ["Term", "Home Ownership", "Purpose"]:  # Categorical columns
#             sample_input[col] = label_encoders[col].classes_[0]
#         else:  # Numeric columns
#             sample_input[col] = 0.0

# # Convert the input into a DataFrame (this ensures correct feature alignment)
# input_df = pd.DataFrame([sample_input])[all_columns]

# label_cols = ['Term', 'Home Ownership', 'Purpose']  # List all categorical columns
# for col in label_cols:
#     if col in sample_input:  # Only apply encoding if the column is present
#         le = label_encoders[col]
#         input_df[col] = le.transform(input_df[col])


# # Make sure to preprocess input (e.g., encoding) if necessary, similar to your FastAPI code
# prediction = model.predict(input_df)  # Adjust as needed based on your input format
# print(prediction)
