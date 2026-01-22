<img width="1920" height="1080" alt="Screenshot 2026-01-22 141029" src="https://github.com/user-attachments/assets/36540ec8-230a-4a4e-b6f9-5c7c9b9c1b60" />
<img width="1920" height="1080" alt="Screenshot 2026-01-22 141047" src="https://github.com/user-attachments/assets/13618a3a-31c0-45c8-947b-625c23a2c938" />
# Federated HeartCare System 

-> Project Overview
The "Federated HeartCare System" is a machine learning–based health monitoring project that predicts heart disease risk using "federated learning".  
Instead of training a single centralized model, multiple user-specific models (Typical, Athletic, Diver) are trained locally and aggregated to preserve data privacy.

This project also includes "concept drift detection" and "dynamic model swapping" to adapt to changes in a user's lifestyle or health condition.

-> Objectives
- Predict heart disease risk accurately
- Preserve user data privacy using federated learning
- Detect behavioral or physiological drift in heart rate data
- Dynamically switch models based on detected drift
- Evaluate system performance before and after adaptation

-> System Architecture
- Local Clients: Typical, Athletic, Diver users
- Local Models: Trained independently on user-specific data
- Federated Server: Aggregates model parameters
- Drift Detector: Identifies lifestyle or health changes
- Model Swapper: Activates the most suitable model

-> Technologies Used
- Programming Language: Python  
- Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Joblib  
- Machine Learning: Logistic Regression  
- Concepts: Federated Learning, Drift Detection  
- Version Control: Git & GitHub  

-> Project Structure
Federated_HeartCare_Project/
│
├── module1_data_preparation.py
├── module2_centralized_model.py
├── module3_federated_learning.py
├── module4_drift_detection.py
├── module5_model_swapping.py
├── module6_evaluation.py
│
├── typical.csv
├── athletic.csv
├── diver.csv
│
├── model_typical.pkl
├── model_athletic.pkl
├── model_diver.pkl
│
└── README.md

-> Modules Description
- Module 1: User data simulation and preparation  
- Module 2: Centralized model training and evaluation  
- Module 3: Federated learning with multiple clients  
- Module 4: Drift detection in heart rate data  
- Module 5: Model swapping based on detected drift  
- Module 6: Performance comparison and visualization
  
-> Key Features
- Privacy-preserving learning
- User-specific model training
- Adaptive system behavior
- Improved accuracy after drift handling
- Real-world healthcare applicability

-> How to Run the Project
 bash
pip install pandas numpy scikit-learn matplotlib joblib
python module1_data_preparation.py
python module2_centralized_model.py
python module3_federated_learning.py
python module4_drift_detection.py
python module5_model_swapping.py
python module6_evaluation.py

-> Output

Trained models for each user type

Drift detection alerts

Accuracy comparison graphs

Adaptive model performance improvement
