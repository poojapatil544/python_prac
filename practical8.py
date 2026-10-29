
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create a dataset
data = {
    "StudyHours": [1, 2, 3, 4, 5, 6],
    "Marks": [35, 45, 50, 65, 75, 90],
    "Attendance": [60, 65, 70, 80, 85, 95]
}

df = pd.DataFrame(data)

print("Dataset:\n", df)

# 1. Scatter Plot
plt.figure()
plt.scatter(df["StudyHours"], df["Marks"])
plt.title("Study Hours vs Marks")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.grid(True)
plt.show()

# 2. Calculate Correlation
correlation = df.corr(numeric_only=True)

print("\nCorrelation Matrix:\n", correlation)

# 3. Correlation Heatmap
plt.figure()
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)
plt.title("Correlation Heatmap")
plt.show()
