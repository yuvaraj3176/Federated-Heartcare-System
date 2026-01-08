import joblib

models = {
    "Typical": joblib.load("model_typical.pkl"),
    "Athletic": joblib.load("model_athletic.pkl"),
    "Diver": joblib.load("model_diver.pkl")
}

current_state = "Typical"

def swap_model(new_state):
    global current_state
    current_state = new_state
    print(f"🔁 Model swapped to {new_state}")

# Example drift response
swap_model("Athletic")
