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
            print("Missing values dropped.")
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


def decision_tree(x_train, x_test, y_train, y_test):
    decision_tree_model = DecisionTreeClassifier(random_state=42)  # Set the random state
    decision_tree_model.fit(x_train, y_train)  # Train the model
    decision_tree_predictions = decision_tree_model.predict(x_test)  # Make predictions
    print("Decision Tree model trained.")
    return decision_tree_model, decision_tree_predictions  # Return the model and predictions


def k_nearest_neighbors(x_train, x_test, y_train, y_test):
    # Create the k-Nearest Neighbors model
    k_nearest_neighbors_model = KNeighborsClassifier() # Set the number of neighbors
    k_nearest_neighbors_model.fit(x_train, y_train.values.ravel()) # Train the model
    k_nearest_neighbors_predictions = k_nearest_neighbors_model.predict(x_test) # Make predictions
    print("k-Nearest Neighbors model trained.")
    return k_nearest_neighbors_predictions # Return the predictions

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

def naive_bayes(x_train, x_test, y_train, y_test):
    # Create the Naïve Bayes model
    naive_bayes_model = GaussianNB() # Gaussian Naïve Bayes
    naive_bayes_model.fit(x_train, y_train.values.ravel()) # Train the model
    naive_bayes_predictions = naive_bayes_model.predict(x_test) # Make predictions
    print("Naïve Bayes model trained.")
    return naive_bayes_predictions # Return the predictions

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

def compare_missing_value_strategies():
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

    print("\nDropping vs Replacing Missing Values :\n")
    print(df)




# Main function
def main():
    # Load the dataset
    data = load_dataset()

    # Preprocess the data
    x_train, x_test, y_train, y_test = preprocess_data(data, target_columns=["Rain"], mv_technique="drop")
    # x_train, x_test, y_train, y_test = preprocess_data(data, target_columns=["Rain"], mv_technique="replace")


    # Display the preprocessed data
    print("\nTraining features:\n", x_train.head())
    print("\nTraining targets:\n", y_train.head())
    print("\nTesting features:\n", x_test.head())
    print("\nTesting targets:\n", y_test.head())

    # Train and evaluate the Decision Tree model
    decision_tree_model, decision_tree_predictions = decision_tree(x_train, x_test, y_train, y_test)
    decision_tree_metrics = evaluate_model(y_test, decision_tree_predictions, "Decision Tree") # Evaluate the model
    # Train and evaluate the k-Nearest Neighbors model
    k_nearest_neighbors_predictions = k_nearest_neighbors(x_train, x_test, y_train, y_test) # Make predictions
    k_nearest_neighbors_metrics = evaluate_model(y_test, k_nearest_neighbors_predictions, "k-Nearest Neighbors") # Evaluate the model
    # Train and evaluate the Naïve Bayes model
    naive_bayes_predictions = naive_bayes(x_train, x_test, y_train, y_test) # Make predictions
    naive_bayes_metrics = evaluate_model(y_test, naive_bayes_predictions, "Naïve Bayes") # Evaluate the model
    # Store the results for comparison
    all_metrics = {
        "Decision Tree": decision_tree_metrics,
        "k-Nearest Neighbors": k_nearest_neighbors_metrics,
        "Naïve Bayes": naive_bayes_metrics
    }
    results_df = pd.DataFrame(all_metrics, index=["Accuracy", "Precision", "Recall"]) # Create a DataFrame
    print("\nModel Comparison:\n", results_df)  # Display the results
    
    # Visualize the decision tree
    plt.figure(figsize=(10, 10))
    plot_tree(decision_tree_model, feature_names=x_train.columns, class_names=["No Rain", "Rain"], filled=True)
    plt.title("Decision Tree Visualization")
    plt.show()

    # Convert your data to numpy arrays with appropriate data types
    x_train_np = np.array(x_train, dtype=np.float64)
    y_train_np = np.array(y_train, dtype=np.int64)
    x_test_np = np.array(x_test, dtype=np.float64)

    # Convert yTrain to 1D
    y_train_np = y_train_np.ravel()

    #Use your custom_knn function to make predictions
    predictions = custom_knn(x_train_np, y_train_np, x_test_np, k=3)
    # Evaluate the Model
    KNN_accuracy = accuracy_score(y_test, predictions)
    KNN_precision = precision_score(y_test, predictions)
    KNN_recall = recall_score(y_test, predictions)
    print("Custom KNN Model Metrics:")
    print(f"Accuracy: {KNN_accuracy:.2f}")
    print(f"Precision: {KNN_precision:.2f}")
    print(f"Recall: {KNN_recall:.2f}")

    #Experiment with Different k Values
    print("Experiment with Different k Values")
    results = []
    for k in [1, 3, 5, 7, 9]:
        predictions = custom_knn(x_train_np, y_train_np, x_test_np, k=k)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)

        # Append the results for this k
        results.append([k, accuracy, precision, recall])

    # Convert the results into a pandas DataFrame
    results_df = pd.DataFrame(results, columns=['k', 'Accuracy', 'Precision', 'Recall'])

    print(results_df)
    ####** --- Comparison ---**
    ### **Custom KNN vs Sklearn KNN**
    comparison = {
        'Metric': ['Accuracy', 'Precision', 'Recall'],
        'Custom KNN': [KNN_accuracy, KNN_precision, KNN_recall],
        'Sklearn KNN': k_nearest_neighbors_metrics
    }

    comparison_df = pd.DataFrame(comparison)
    print("\nComparison (Custom KNN vs Sklearn KNN):")
    print(comparison_df)
    
    compare_missing_value_strategies()
    


if __name__ == "__main__":
    main()