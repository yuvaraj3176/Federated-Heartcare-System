import numpy as np

# Simple drift detection using threshold
def simple_drift_detector(data, threshold=10):
    baseline = np.mean(data[:50])
    for i, rate in enumerate(data[50:], start=50):
        if abs(rate - baseline) > threshold:
            return i
    return None

heart_rate_stream = np.random.normal(70, 2, 100)

# Simulate lifestyle change
heart_rate_stream[50:] += 20  # Drift introduced

drift_index = simple_drift_detector(heart_rate_stream)
if drift_index:
    print(f"⚠ Drift detected at index {drift_index}")
else:
    print("No drift detected")
