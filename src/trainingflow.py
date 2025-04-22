from metaflow import FlowSpec, step, Parameter
import pandas as pd 
import joblib

class BankLoanTrainFlow(FlowSpec):
    # Set default parameters from registered best model
    n_estimators = Parameter('n_estimators', default = 486)
    max_features = Parameter('max_features', default = 8)

    @step
    def start(self):
        # Load training data
        self.train = pd.read_csv("./bank-loan/credit_train.csv")
        self.next(self.preprocessing)

    @step
    def preprocessing(self):
        # Remove null rows and irrelevant columns
        self.train = self.train.dropna(how='all') 
        self.train = self.train.drop(columns=["Loan ID", "Customer ID"])
        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        from sklearn.preprocessing import LabelEncoder

        # Encode target variable
        self.target_le = LabelEncoder()
        self.train['Loan Status'] = self.target_le.fit_transform(self.train['Loan Status'])

        # Encode other categorical features 
        label_cols = ['Home Ownership', 'Purpose','Years in current job','Term'] # no foreach parallelism required because not row-wise
        self.label_encoders = {}
    
        for col in label_cols:
            le = LabelEncoder()
            self.train[col] = le.fit_transform(self.train[col])
            self.label_encoders[col] = le
         
        self.next(self.model)
    
    @step
    def model(self):
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from sklearn.ensemble import RandomForestClassifier
        import mlflow
        print("MLflow version in Metaflow:", mlflow.__version__)

        # Set tracking uri
        mlflow.set_tracking_uri('sqlite:////Users/shruti/USF-Spring-2/ML-Ops/src/mlflow.db')
        mlflow.set_experiment("bank-loan-experiment")
        
        # Split data into training and validation set 
        X = self.train.drop(columns = "Loan Status")
        y = self.train["Loan Status"]

        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(X,y, test_size= 0.2, random_state= 25)

        # Track using mlflow
        with mlflow.start_run():
            mlflow.log_params({'n_estimators':self.n_estimators, 'max_features':self.max_features})
            self.rf = RandomForestClassifier(n_estimators = self.n_estimators, max_features = self.max_features)
            self.rf.fit(self.x_train,self.y_train)
            y_hat = self.rf.predict(self.x_val)

            accuracy = accuracy_score(self.y_val, y_hat)

            mlflow.set_tags({"Model":"Random Forest"})
            mlflow.log_metric('accuracy', accuracy)
            mlflow.sklearn.log_model(self.rf, artifact_path="bank_model")
            self.run_id = mlflow.active_run().info.run_id
        
        # Print accuracy 
        print("Accuracy:", accuracy)
        self.next(self.end)

    @step
    def end(self):
        # Save label encoders to transform test set 
        joblib.dump(self.rf, "bank_model.pkl")
        joblib.dump(self.target_le, "target_encoder.pkl")
        joblib.dump(self.label_encoders, "feature_encoders.pkl")

        with open("latest_run_id.txt", "w") as f:
            f.write(self.run_id)

        print(f"Saved run_id: {self.run_id}")

if __name__=='__main__':
    BankLoanTrainFlow()
