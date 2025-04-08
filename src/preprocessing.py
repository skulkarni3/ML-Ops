from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
import pandas as pd 

wine = load_wine()
df_wine = pd.DataFrame(data=wine.data, columns=wine.feature_names)
y = wine.target
X = df_wine
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

X_train.to_parquet('data/x_train_by_dvc.parquet')
X_test.to_parquet('data/x_test_by_dvc.parquet')