import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
pd.set_option('future.no_silent_downcasting', True)


# Loading the `weather_forecast_data.csv` dataset
def load_dataset():
    data = pd.read_csv("weather_forecast_data.csv")
    print("Data loaded successfully.")
    return data


# Preprocessing the data
def preprocess_data(data, target_columns, mv_technique):
    # Identify the missing values
    missing_values = data.isna().sum()
    print("Missing values:\n", missing_values)

    # Handle the missing values
    if missing_values.any():
        # Apply first technique (drop missing values)
        if mv_technique == "drop":
            data = data.dropna(axis=0)
            print("\nMissing values dropped.")
        # Apply second technique (replace missing values)
        elif mv_technique == "replace":
            for column in data.columns:
                if data[column].isna().any():
                    data.fillna({column: data[column].mean()}, inplace=True)
            print("Missing values imputed.")
    else:
        print("No missing values found.")

    # Separate the data into features and targets
    features = data.drop(columns=target_columns)
    targets = data[target_columns]
    print("Features and targets are separated.")

    # Encode target labels as binary (0 and 1)
    targets = targets.replace({'no rain': 0, 'rain': 1}).astype(int)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42, shuffle=True)
    print("Data is shuffled and split.")

    # Scale the features after applying the train-test split
    scaler = StandardScaler()
    numerical_columns = x_train.select_dtypes(include=[np.number]).columns
    x_train[numerical_columns] = scaler.fit_transform(x_train[numerical_columns])
    x_test[numerical_columns] = scaler.transform(x_test[numerical_columns])
    print("Features are scaled.")

    print("Preprocessing completed!")
    return x_train, x_test, y_train, y_test


# Decision tree model using scikit-learn library
def decision_tree(x_train, x_test, y_train):
    decision_tree_model = DecisionTreeClassifier(random_state=42)  # Set the random state
    decision_tree_model.fit(x_train, y_train)  # Train the model
    decision_tree_predictions = decision_tree_model.predict(x_test)  # Make predictions
    return decision_tree_model, decision_tree_predictions  # Return the model and predictions


# k-Nearest Neighbors model using scikit-learn library
def k_nearest_neighbors(x_train, x_test, y_train):
    k_nearest_neighbors_model = KNeighborsClassifier() # Set the number of neighbors
    k_nearest_neighbors_model.fit(x_train, y_train.values.ravel()) # Train the model
    k_nearest_neighbors_predictions = k_nearest_neighbors_model.predict(x_test) # Make predictions
    return k_nearest_neighbors_predictions # Return the predictions


# Naïve Bayes model using scikit-learn library
def naive_bayes(x_train, x_test, y_train):
    naive_bayes_model = GaussianNB() # Gaussian Naïve Bayes
    naive_bayes_model.fit(x_train, y_train.values.ravel()) # Train the model
    naive_bayes_predictions = naive_bayes_model.predict(x_test) # Make predictions
    return naive_bayes_predictions # Return the predictions


# Custom k-Nearest Neighbors model
def custom_knn(xTrain, yTrain, xTest, k):
    predictions = []
    for i in range(xTest.shape[0]):  # Iterate over test samples
        test_point = xTest[i]  # Single test point

        # Calculate distances from test_point to all points in xTrain
        distances = np.sqrt(np.sum((xTrain - test_point) ** 2, axis=1))

        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:k]
        k_labels = yTrain[k_indices]  # Fetch their corresponding labels

        # Majority vote (classification)
        prediction = np.bincount(k_labels).argmax()
        predictions.append(prediction)
    return predictions


# Evaluate models using accuracy, precision, and recall
def evaluate_model(y_test, predictions, model_name):
    # Calculate the accuracy, precision, and recall
    accuracy = accuracy_score(y_test, predictions) # Compare the predicted values to the actual values
    precision = precision_score(y_test, predictions,average='binary') # Compare the predicted positive values to the actual positive values
    recall = recall_score(y_test, predictions,average='binary') # Compare the predicted positive values to the actual positive values
    print(f"\n{model_name} Model Metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    return accuracy, precision, recall # Return the metrics


# Compare the results of the two missing values handling techniques
def compare_missing_data_techniques():
    data = {
        "Model": ["Decision Tree", "k-Nearest Neighbors", "Naïve Bayes"] * 2,
        "Strategy": ["Dropping"] * 3 + ["Replacing"] * 3,
        "Accuracy": [0.997872340425532, 0.9617021276595744, 0.9617021276595744,
                     0.996, 0.968, 0.964],
        "Precision": [1.0, 0.890625, 1.0, 1.0, 0.9166666666666666, 1.0],
        "Recall": [0.9852941176470589, 0.8382352941176471, 0.7352941176470589,
                   0.9642857142857143, 0.7857142857142857, 0.6785714285714286]
    }
    df = pd.DataFrame(data)
    print("\nComparison Between Different Missing Values Handling Techniques (Dropping vs Replacing):\n", df)


# Visualize the decision tree
def visualize_tree(decision_tree_model, x_train):
    plt.figure(figsize=(10, 10))
    plot_tree(decision_tree_model, feature_names=x_train.columns, class_names=["No Rain", "Rain"], filled=True)
    plt.title("Decision Tree Visualization")
    plt.show()


# Train and evaluate the custom KNN model
def train_and_evaluate_custom_knn(x_train, x_test, y_train, y_test):
    # Train the model to make predictions
    predictions = custom_knn(x_train, y_train, x_test, k=3)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)

    print("\nCustom KNN Model Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    return accuracy, precision, recall


# Experiment with different k Values
def try_different_k(x_train, x_test, y_train, y_test):
    results = []
    for k in [1, 3, 5, 7, 9]:
        predictions = custom_knn(x_train, y_train, x_test, k=k)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)

        # Append the results for this k
        results.append([k, accuracy, precision, recall])

    # Convert the results into a pandas DataFrame
    results_df = pd.DataFrame(results, columns=['k', 'Accuracy', 'Precision', 'Recall'])
    return results_df


# Main function
def main():
    # Load the dataset
    data = load_dataset()

    # ---- Task #1 ----
    # Preprocess the data (using the "drop missing values" technique)
    x_train, x_test, y_train, y_test = preprocess_data(data, target_columns=["Rain"], mv_technique="drop")

    # ---- Task #2 ----
    # 2.1 Train scikit-learn models
    # Decision Tree model
    decision_tree_model, decision_tree_predictions = decision_tree(x_train, x_test, y_train)
    decision_tree_metrics = evaluate_model(y_test, decision_tree_predictions, "Decision Tree") # Evaluate the model

    # k-Nearest Neighbors model
    k_nearest_neighbors_predictions = k_nearest_neighbors(x_train, x_test, y_train) # Make predictions
    k_nearest_neighbors_metrics = evaluate_model(y_test, k_nearest_neighbors_predictions, "k-Nearest Neighbors") # Evaluate the model

    # Naïve Bayes model
    naive_bayes_predictions = naive_bayes(x_train, x_test, y_train) # Make predictions
    naive_bayes_metrics = evaluate_model(y_test, naive_bayes_predictions, "Naïve Bayes") # Evaluate the model

    # 2.2 Evaluate and store the results for comparison
    all_metrics = {
        "Decision Tree": decision_tree_metrics,
        "k-Nearest Neighbors": k_nearest_neighbors_metrics,
        "Naïve Bayes": naive_bayes_metrics
    }
    results_df = pd.DataFrame(all_metrics, index=["Accuracy", "Precision", "Recall"]) # Create a DataFrame
    print("\nModel Comparison:\n", results_df)  # Display the results

    # 2.3 Implement a custom k-Nearest Neighbors model from scratch
    # Convert your data to numpy arrays with appropriate data types
    x_train_np = np.array(x_train, dtype=np.float64)
    y_train_np = np.array(y_train, dtype=np.int64)
    x_test_np = np.array(x_test, dtype=np.float64)

    # Convert yTrain to 1D
    y_train_np = y_train_np.ravel()

    # Train and evaluate the custom KNN model
    KNN_accuracy, KNN_precision, KNN_recall = train_and_evaluate_custom_knn(x_train_np, x_test_np, y_train_np, y_test)

    # 2.4 Comparsion between custom KNN and scikit-learn's KNN
    comparison = {
        'Metric': ['Accuracy', 'Precision', 'Recall'],
        'Custom KNN': [KNN_accuracy, KNN_precision, KNN_recall],
        'scikit-learn KNN': k_nearest_neighbors_metrics
    }
    comparison_df = pd.DataFrame(comparison)
    print("\nComparison (Custom KNN vs scikit-learn KNN):\n", comparison_df)

    # ---- Task #3 ----
    # 3.1 Compare the results of the two missing values handling techniques
    compare_missing_data_techniques()

    # 3.2 Visualization of Decision Tree
    visualize_tree(decision_tree_model, x_train)

    # 3.3 Performace metrics for different k values
    # Experiment with different k values
    results_of_diff_k = try_different_k(x_train_np, x_test_np, y_train_np, y_test)

    # Merge custom knn results and scikit-learn's knn results into one DataFrame
    sklearn_knn_metrics_df = pd.DataFrame([k_nearest_neighbors_metrics], columns=['Accuracy', 'Precision', 'Recall'])
    sklearn_knn_metrics_df['k'] = 'scikit-learn KNN'
    
    # Append the k-nearest neighbors metrics to the results of different k values
    merged_results = pd.concat([results_of_diff_k, sklearn_knn_metrics_df], ignore_index=True)
    
    # Print the merged results
    print("\nPerformance Comparison (Different k values and scikit-learn KNN):\n", merged_results)


if __name__ == "__main__":
    main()