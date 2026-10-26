"""
Project 2: Data Classification Using AI
---------------------------------------
Trains a Decision Tree classifier on the Iris dataset to predict flower
species from petal and sepal measurements, then evaluates its performance.
"""

# ---------------------------------------------------------------------------
# STEP 0: Import libraries
# ---------------------------------------------------------------------------
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------------------
# STEP 1: Load the dataset
# Iris: 150 samples, 4 features, 3 flower species (classes)
# ---------------------------------------------------------------------------
iris = load_iris()
X = iris.data    # Features: sepal length/width, petal length/width (in cm)
y = iris.target  # Labels: 0 = setosa, 1 = versicolor, 2 = virginica

print("Dataset loaded successfully")
print("Total samples   :", X.shape[0])
print("Features        :", iris.feature_names)
print("Classes         :", iris.target_names.tolist())

# ---------------------------------------------------------------------------
# STEP 2: Split into training and testing sets
# 80% of the data trains the model, 20% is held back to test it on
# unseen examples. random_state makes the split reproducible, and
# stratify keeps the class proportions equal in both sets.
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

# ---------------------------------------------------------------------------
# STEP 3: Train the model
# A Decision Tree learns a series of if/else rules from the training data.
# max_depth limits the tree size to reduce overfitting.
# ---------------------------------------------------------------------------
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

print("\nModel trained successfully")

# ---------------------------------------------------------------------------
# STEP 4: Test the model (make predictions on unseen data)
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test)

# ---------------------------------------------------------------------------
# STEP 5: Evaluate the model
# accuracy_score        -> fraction of predictions that were correct
# classification_report -> precision, recall and F1-score for each class
# ---------------------------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ---------------------------------------------------------------------------
# STEP 6 (Bonus): Predict a single new flower
# Measurements: sepal length, sepal width, petal length, petal width (cm)
# ---------------------------------------------------------------------------
new_flower = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(new_flower)
print("Prediction for", new_flower[0], "->", iris.target_names[prediction[0]])
