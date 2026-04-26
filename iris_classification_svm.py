from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def main():
    # Load Iris dataset
    iris = datasets.load_iris()

    X = iris.data       # features
    y = iris.target     # labels

    classes = iris.target_names

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Create SVM model
    model = svm.SVC()

    # Train model
    model.fit(X_train, y_train)

    # Predict on test data
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)

    print("Iris Flower Classification using SVM")
    print("-----------------------------------")
    print(f"Accuracy: {accuracy:.2f}")
    print()

    # Show prediction vs actual
    for pred, actual in zip(predictions, y_test):
        print(f"Predicted: {classes[pred]}")
        print(f"Actual:    {classes[actual]}")
        print("-" * 30)

    # Full report
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=classes))


if __name__ == "__main__":
    main()