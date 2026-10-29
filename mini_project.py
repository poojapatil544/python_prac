
# ALL 11 DATA SCIENCE PRACTICALS
# NumPy, Pandas, Matplotlib, Seaborn, SciPy, Scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score


# PRACTICAL 1: NumPy Arrays, Indexing, Slicing and Reshaping
def practical1():
    print("\n========== PRACTICAL 1 ==========")

    arr = np.array([10, 20, 30, 40, 50])
    print("Original Array:", arr)
    print("Element at index 2:", arr[2])
    print("Slicing:", arr[1:4])

    matrix = np.arange(1, 7).reshape(2, 3)
    print("Reshaped Array:\n", matrix)

    print("Addition:", arr + 5)
    print("Multiplication:", arr * 2)
    print("Square:", arr ** 2)


# PRACTICAL 2: Matrix and Statistical Operations
def practical2():
    print("\n========== PRACTICAL 2 ==========")

    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    print("Matrix A:\n", A)
    print("Matrix B:\n", B)
    print("Addition:\n", A + B)
    print("Subtraction:\n", A - B)
    print("Matrix Multiplication:\n", np.dot(A, B))
    print("Transpose of A:\n", A.T)

    data = np.array([10, 20, 30, 40, 50])
    print("Mean:", np.mean(data))
    print("Median:", np.median(data))
    print("Variance:", np.var(data))
    print("Standard Deviation:", np.std(data))
    print("Minimum:", np.min(data))
    print("Maximum:", np.max(data))


# PRACTICAL 3: Pandas Data Import, Inspection and Statistics
def practical3():
    print("\n========== PRACTICAL 3 ==========")

    data = {
        "Name": ["Amit", "Priya", "Rahul", "Sneha", "Neha"],
        "Age": [20, 21, 19, 22, 20],
        "Marks": [85, 90, 78, 92, 88]
    }

    df = pd.DataFrame(data)
    df.to_csv("students.csv", index=False)
    df = pd.read_csv("students.csv")

    print("Dataset:\n", df)
    print("First Five Rows:\n", df.head())
    print("Dataset Information:")
    df.info()
    print("Rows and Columns:", df.shape)
    print("Column Names:", df.columns.tolist())
    print("Descriptive Statistics:\n", df.describe())


# PRACTICAL 4: Data Preprocessing
def practical4():
    print("\n========== PRACTICAL 4 ==========")

    data = {
        "Name": ["Amit", "Priya", "Rahul", "Priya"],
        "Age": [20, np.nan, 19, np.nan],
        "Marks": [85, 90, 78, 90]
    }

    df = pd.DataFrame(data)
    print("Original Dataset:\n", df)

    df["Age"] = df["Age"].fillna(df["Age"].mean())
    df = df.drop_duplicates()
    df["Result"] = df["Marks"].apply(
        lambda x: "Pass" if x >= 40 else "Fail"
    )
    df = df.rename(columns={"Marks": "Score"})

    print("Preprocessed Dataset:\n", df)
    print("Missing Values:\n", df.isnull().sum())


# PRACTICAL 5: Filtering, Sorting and Selection
def practical5():
    print("\n========== PRACTICAL 5 ==========")

    data = {
        "Name": ["Amit", "Priya", "Rahul", "Sneha", "Neha"],
        "Marks": [75, 92, 35, 88, 65],
        "Age": [20, 21, 19, 22, 20]
    }

    df = pd.DataFrame(data)

    print("Original Dataset:\n", df)
    print("Marks Above 70:\n", df[df["Marks"] > 70])
    print("Passed Students:\n",
          df[(df["Marks"] >= 40) & (df["Age"] >= 20)])
    print("Sorted by Marks:\n",
          df.sort_values(by="Marks", ascending=False))
    print("Selected Columns:\n", df[["Name", "Marks"]])


# PRACTICAL 6: Grouping and Aggregation
def practical6():
    print("\n========== PRACTICAL 6 ==========")

    data = {
        "Department": [
            "Computer", "IT", "Computer",
            "IT", "Electronics", "Computer"
        ],
        "Student": ["Amit", "Priya", "Rahul",
                    "Sneha", "Neha", "Raj"],
        "Marks": [85, 90, 78, 88, 92, 80]
    }

    df = pd.DataFrame(data)
    print("Original Dataset:\n", df)

    print("Average Marks:\n",
          df.groupby("Department")["Marks"].mean())

    print("Department Statistics:\n",
          df.groupby("Department")["Marks"].agg(
              ["mean", "sum", "min", "max", "count"]
          ))

    print("Student Count:\n",
          df.groupby("Department")["Student"].count())


# PRACTICAL 7: Line Plot, Bar Chart and Histogram
def practical7():
    print("\n========== PRACTICAL 7 ==========")

    months = ["Jan", "Feb", "Mar", "Apr", "May"]
    sales = [100, 150, 120, 180, 200]

    plt.figure()
    plt.plot(months, sales, marker="o")
    plt.title("Monthly Sales - Line Plot")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.bar(months, sales)
    plt.title("Monthly Sales - Bar Chart")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.tight_layout()
    plt.show()

    marks = [45, 55, 60, 65, 70, 72, 75, 80, 85, 90, 95]

    plt.figure()
    plt.hist(marks, bins=5, edgecolor="black")
    plt.title("Distribution of Student Marks")
    plt.xlabel("Marks")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


# PRACTICAL 8: Scatter Plot and Correlation Heatmap
def practical8():
    print("\n========== PRACTICAL 8 ==========")

    data = {
        "StudyHours": [1, 2, 3, 4, 5, 6],
        "Marks": [35, 45, 50, 65, 75, 90],
        "Attendance": [60, 65, 70, 80, 85, 95]
    }

    df = pd.DataFrame(data)
    print("Dataset:\n", df)

    plt.figure()
    plt.scatter(df["StudyHours"], df["Marks"])
    plt.title("Study Hours vs Marks")
    plt.xlabel("Study Hours")
    plt.ylabel("Marks")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    correlation = df.corr(numeric_only=True)
    print("Correlation Matrix:\n", correlation)

    plt.figure()
    sns.heatmap(correlation, annot=True, cmap="coolwarm",
                vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


# PRACTICAL 9: Statistical Analysis
def practical9():
    print("\n========== PRACTICAL 9 ==========")

    x = np.array([10, 20, 30, 40, 50])
    y = np.array([15, 25, 35, 45, 55])

    print("Mean:", np.mean(x))
    print("Median:", np.median(x))
    print("Variance:", np.var(x))
    print("Standard Deviation:", np.std(x))

    correlation, p_value = pearsonr(x, y)
    print("Correlation Coefficient:", correlation)
    print("P-value:", p_value)


# PRACTICAL 10: Simple Linear Regression
def practical10():
    print("\n========== PRACTICAL 10 ==========")

    X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.array([35, 40, 50, 60, 70])

    model = LinearRegression()
    model.fit(X, y)

    prediction = model.predict([[6]])

    print("Slope:", model.coef_[0])
    print("Intercept:", model.intercept_)
    print("Predicted Marks:", prediction[0])

    predicted_values = model.predict(X)
    print("Mean Squared Error:",
          mean_squared_error(y, predicted_values))


# PRACTICAL 11: Decision Tree Classification
def practical11():
    print("\n========== PRACTICAL 11 ==========")

    X = np.array([
        [1, 40], [2, 45], [3, 50], [4, 60],
        [5, 70], [6, 80], [7, 90], [8, 95]
    ])

    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)

    new_student = np.array([[6, 75]])
    prediction = model.predict(new_student)

    if prediction[0] == 1:
        print("Predicted Result: Pass")
    else:
        print("Predicted Result: Fail")

    predicted_values = model.predict(X)
    accuracy = accuracy_score(y, predicted_values)
    print("Training Accuracy:", accuracy * 100, "%")


# MAIN PROGRAM
if __name__ == "__main__":
    practical1()
    practical2()
    practical3()
    practical4()
    practical5()
    practical6()
    practical7()
    practical8()
    practical9()
    practical10()
    practical11()

    print("\nAll 11 practicals executed successfully!")
