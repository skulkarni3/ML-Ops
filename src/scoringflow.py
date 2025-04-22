from metaflow import FlowSpec, step
import pandas as pd 
import joblib

class BankLoansPredictFlow(FlowSpec):
    @step
    def start(self):
        # Load testing data 
        self.test = pd.read_csv("./bank-loan/credit_test.csv")
        # Load latest run_id
        with open("latest_run_id.txt", "r") as f:
            self.run_id = f.read().strip()
    
        print(f"Using model from run_id: {self.run_id}")

        self.next(self.preprocessing)

    @step
    def preprocessing(self):
        # Remove null rows and irrelevant columns
        self.test = self.test.dropna(how='all') 
        self.test = self.test.drop(columns=["Loan ID", "Customer ID"])
        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        # Load label encoders fit on training data
        self.label_encoders = joblib.load("feature_encoders.pkl")

        # Encode categorical features (since test set has no target variable)
        label_cols = ['Home Ownership', 'Purpose','Years in current job','Term'] 
    
        for col in label_cols:
            le = self.label_encoders[col]
            self.test[col] = le.transform(self.test[col])  
         
        self.next(self.load_model)

    @step
    def load_model(self):
        import mlflow
        # Load model from mlflow
        mlflow.set_tracking_uri("sqlite:////Users/shruti/USF-Spring-2/ML-Ops/src/mlflow.db")
        bank_model = f"runs:/{self.run_id}/bank_model"
        self.model = mlflow.sklearn.load_model(bank_model)
        self.next(self.predict)

    @step 
    def predict(self):
        # Load target encoder to decode predictions
        self.target_le = joblib.load("target_encoder.pkl")
        predictions = self.model.predict(self.test)
        self.decoded_preds = pd.Series(self.target_le.inverse_transform(predictions))
        self.next(self.end)

    @step
    def end(self):
        # Print and save predictions to csv
        print(f"Predictions:\n{self.decoded_preds.value_counts()}")
        self.predictions_df = pd.DataFrame(self.decoded_preds, columns=['Predicted Loan Status'])
        self.predictions_df.to_csv("predictions.csv", index=False)

if __name__ == '__main__':
    BankLoansPredictFlow()


