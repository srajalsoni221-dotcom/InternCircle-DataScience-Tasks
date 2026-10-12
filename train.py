from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load data and train model
iris = load_iris()
clf = DecisionTreeClassifier(random_state=42)
clf.fit(iris.data, iris.target)

# Save the compatible model locally
joblib.dump(clf, 'iris_classifier_model.pkl')
print("Local model saved successfully!")
