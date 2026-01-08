import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

clients = {
    "Typical": pd.read_csv("typical.csv"),
    "Athletic": pd.read_csv("athletic.csv"),
    "Diver": pd.read_csv("diver.csv")
}

global_weights = None

def train_local_model(client_data, client_name):
    # Encode categorical variables
    categorical_cols = ['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    client_data = pd.get_dummies(client_data, columns=categorical_cols, drop_first=True)
    
    X = client_data.drop(["num", "user_type", "id"], axis=1)
    y = client_data["num"]
    
    # Impute missing values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    # Save the model
    joblib.dump(model, f"model_{client_name.lower()}.pkl")

    return model.coef_, model.intercept_

# Federated Averaging
for round in range(3):
    print(f"\n🔄 Federated Round {round + 1}")
    weights, biases = [], []

    for name, data in clients.items():
        w, b = train_local_model(data, name)
        weights.append(w)
        biases.append(b)
        print(f"Client {name} trained locally")

    global_weights = np.mean(weights, axis=0)
    global_bias = np.mean(biases, axis=0)

print("✔ Federated Learning Completed")
