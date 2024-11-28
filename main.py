import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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


main()
