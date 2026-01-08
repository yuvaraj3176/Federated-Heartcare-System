import matplotlib.pyplot as plt

accuracy_before = [0.82, 0.81, 0.80]
accuracy_after = [0.82, 0.88, 0.90]

plt.plot(accuracy_before, label="Before Drift")
plt.plot(accuracy_after, label="After Model Swap")
plt.xlabel("Time")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Federated HeartCare Performance")
plt.show()
