import pandas as pd
import numpy as np
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Step 1: Data Preparation
def load_data(file_path, data_percentage, test_size):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Take the required percentage of data
    data = data.sample(frac=data_percentage)

    # Split data into features and labels
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Labels

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test, X.columns


# Step 2: Implement Bayesian Classifier
class BayesianClassifier:
    def __init__(self):
        pass

    def fit(self, X, y):
        # Implement Bayesian classifier training
        self.class_probabilities = {}
        self.feature_probabilities = {}

        # Calculate class probabilities
        class_counts = y.value_counts()
        total_samples = len(y)
        for class_label, count in class_counts.items():
            self.class_probabilities[class_label] = count / total_samples

        # Calculate feature probabilities
        for feature in X.columns:
            self.feature_probabilities[feature] = {}
            for feature_value in X[feature].unique():
                self.feature_probabilities[feature][feature_value] = {}
                for class_label in self.class_probabilities.keys():
                    class_mask = (y == class_label)
                    feature_mask = (X[feature] == feature_value)
                    count = np.logical_and(class_mask, feature_mask).sum()
                    total_class_count = class_mask.sum()
                    self.feature_probabilities[feature][feature_value][class_label] = count / total_class_count

    def predict(self, X):
        # Implement prediction using Bayesian classifier
        predictions = []
        for _, sample in X.iterrows():
            max_probability = -1
            predicted_class = None
            for class_label, class_probability in self.class_probabilities.items():
                probability = class_probability
                for feature, value in sample.items():
                    if value in self.feature_probabilities[feature]:
                        probability *= self.feature_probabilities[feature][value][class_label]
                    else:
                        probability *= 0  # Laplace smoothing
                if probability > max_probability:
                    max_probability = probability
                    predicted_class = class_label
            predictions.append(predicted_class)
        return predictions


# Step 3: Implement Decision Tree Classifier
class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _calculate_gini(self, y):
        # Calculate Gini impurity
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def _find_best_split(self, X, y):
        best_gini = float('inf')
        best_split_feature = None
        best_split_value = None

        for feature in X.columns:
            for value in X[feature].unique():
                left_mask = X[feature] <= value
                right_mask = ~left_mask
                left_gini = self._calculate_gini(y[left_mask])
                right_gini = self._calculate_gini(y[right_mask])
                total_gini = (len(y[left_mask]) * left_gini + len(y[right_mask]) * right_gini) / len(y)
                if total_gini < best_gini:
                    best_gini = total_gini
                    best_split_feature = feature
                    best_split_value = value

        return best_split_feature, best_split_value

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return {'prediction': y.iloc[0]}

        if X.empty:
            return {'prediction': y.value_counts().idxmax()}

        best_feature, best_value = self._find_best_split(X, y)
        left_mask = X[best_feature] <= best_value
        right_mask = ~left_mask
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {'feature': best_feature, 'value': best_value,
                'left': left_tree, 'right': right_tree}

    def predict(self, X):
        # Implement prediction using Decision Tree classifier
        predictions = []
        for _, sample in X.iterrows():
            node = self.tree
            while 'prediction' not in node:
                if sample[node['feature']] <= node['value']:
                    node = node['left']
                else:
                    node = node['right']
            predictions.append(node['prediction'])
        return predictions


# Function to handle file selection
def browse_file():
    filename = filedialog.askopenfilename()
    entry.delete(0, END)
    entry.insert(0, filename)


# Function to handle classification
def classify():
    file_path = entry.get()
    data_percentage = float(data_percentage_entry.get()) / 100
    training_size = float(training_size_entry.get()) / 100
    test_size = float(test_size_entry.get()) / 100
    try:
        X_train, X_test, y_train, y_test, attributes = load_data(file_path, data_percentage, test_size)

        # Further split the training data to get training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=(1 - training_size),
                                                          random_state=42)

        # Print the shapes of the resulting datasets
        print("Training set shape:", X_train.shape, y_train.shape)
        print("Validation set shape:", X_val.shape, y_val.shape)
        print("Test set shape:", X_test.shape, y_test.shape)

        bayesian_classifier = BayesianClassifier()
        bayesian_classifier.fit(X_train, y_train)
        decision_tree_classifier = DecisionTreeClassifier(max_depth=5)
        decision_tree_classifier.fit(X_train, y_train)

        bayesian_predictions = bayesian_classifier.predict(X_test)
        bayesian_accuracy = accuracy_score(y_test, bayesian_predictions)

        decision_tree_predictions = decision_tree_classifier.predict(X_test)
        decision_tree_accuracy = accuracy_score(y_test, decision_tree_predictions)

        messagebox.showinfo("Results",
                            f"Bayesian Classifier Accuracy: {bayesian_accuracy}\nDecision Tree Classifier Accuracy: {decision_tree_accuracy}")

        # Display the predicted class labels with attributes
        prediction_text.delete(1.0, END)
        prediction_text.insert(END, "Predicted Class Labels with Attributes:\n")
        for i, prediction in enumerate(bayesian_predictions):
            prediction_text.insert(END, f"Attributes: {X_test.iloc[i].values}, Predicted Class: {prediction}\n")

        # Print the number of output classified records
        print("Number of Output Classified Records:", len(bayesian_predictions))

    except Exception as e:
        messagebox.showerror("Error", str(e))



# Step 4: GUI Setup
root = Tk()
root.title("Diabetes Classifier")

label = Label(root, text="Select File:")
label.grid(row=0, column=0)

entry = Entry(root, width=50)
entry.grid(row=0, column=1, padx=10)

browse_button = Button(root, text="Browse", command=browse_file)
browse_button.grid(row=0, column=2)

data_percentage_label = Label(root, text="Data Percentage (%):")
data_percentage_label.grid(row=1, column=0)

data_percentage_entry = Entry(root, width=10)
data_percentage_entry.grid(row=1, column=1, padx=10)

training_size_label = Label(root, text="Training Size (%):")
training_size_label.grid(row=2, column=0)

training_size_entry = Entry(root, width=10)
training_size_entry.grid(row=2, column=1, padx=10)

test_size_label = Label(root, text="Test Size (%):")
test_size_label.grid(row=3, column=0)

test_size_entry = Entry(root, width=10)
test_size_entry.grid(row=3, column=1, padx=10)

classify_button = Button(root, text="Classify", command=classify)
classify_button.grid(row=4, column=1, pady=10)

prediction_text = Text(root, height=30, width=100)
prediction_text.grid(row=5, columnspan=3, padx=10, pady=10)

root.mainloop()
