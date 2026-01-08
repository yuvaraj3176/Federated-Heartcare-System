import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Load merged dataset
data = pd.concat([
    pd.read_csv("typical.csv"),
    pd.read_csv("athletic.csv"),
    pd.read_csv("diver.csv")
])

# Encode categorical variables
categorical_cols = ['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

X = data.drop(["num", "user_type", "id"], axis=1)
y = data["num"]

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Centralized Model Accuracy:", accuracy)
