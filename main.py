import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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
    # Create the Decision Tree model
    decision_tree_model = DecisionTreeClassifier(random_state=42) # Set the random state for reproducibility
    decision_tree_model.fit(x_train, y_train) # Train the model
    decision_tree_predictions = decision_tree_model.predict(x_test) # Make predictions
    print("Decision Tree model trained.")
    return decision_tree_predictions # Return the predictions

def k_nearest_neighbors(x_train, x_test, y_train, y_test):
    # Create the k-Nearest Neighbors model
    k_nearest_neighbors_model = KNeighborsClassifier() # Set the number of neighbors
    k_nearest_neighbors_model.fit(x_train, y_train.values.ravel()) # Train the model
    k_nearest_neighbors_predictions = k_nearest_neighbors_model.predict(x_test) # Make predictions
    print("k-Nearest Neighbors model trained.")
    return k_nearest_neighbors_predictions # Return the predictions

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


# Main function
def main():
    # Load the dataset
    data = load_dataset()

    # Preprocess the data
    x_train, x_test, y_train, y_test = preprocess_data(data, target_columns=["Rain"], mv_technique="replace")

    # Display the preprocessed data
    print("\nTraining features:\n", x_train.head())
    print("\nTraining targets:\n", y_train.head())
    print("\nTesting features:\n", x_test.head())
    print("\nTesting targets:\n", y_test.head())

    # Train and evaluate the Decision Tree model
    decision_tree_predictions = decision_tree(x_train, x_test, y_train, y_test) # Make predictions
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

if __name__ == "__main__":
    main()