import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# TODO: load the cencus.csv data
project_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(project_path, "data", "census.csv")
data = pd.read_csv(data_path)
print(data_path)


label = "salary"

# TODO: split the provided data to have a train dataset and a test dataset
train_data, test_data = train_test_split(
    data,
    test_size=0.20,
    random_state=42,
    stratify=data[label],
)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# TODO: use the process_data function provided to process the data.
X_train, y_train, encoder, lb = process_data(
    X=train_data,
    categorical_features=cat_features,
    label=label,
    training=True,
)

X_test, y_test, _, _ = process_data(
    test_data,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# TODO: use the train_model function to train the model on the training dataset
model = train_model(X_train, y_train)
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

print("Precision:", precision)
print("Recall:", recall)
print("F-beta:", fbeta)

os.makedirs(os.path.join(project_path, "model"), exist_ok=True)

# save the model and the encoder
model_path = os.path.join(project_path, "model", "model.pkl")
save_model(model, model_path)
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
save_model(encoder, encoder_path)

# load the model
lb_path = os.path.join(project_path, "model", "lb.pkl")
save_model(lb, lb_path)
model = load_model(model_path)
encoder = load_model(encoder_path)
lb = load_model(lb_path)

# TODO: use the inference function to run the model inferences
# on the test dataset.
preds = inference(model, X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# TODO: compute the performance on model slices using the
# performance_on_categorical_slice function
# iterate through the categorical features
for col in cat_features:
    # iterate through the unique values in one categorical feature
    for slicevalue in sorted(test_data[col].unique()):
        count = test_data[test_data[col] == slicevalue].shape[0]
        p, r, fb = performance_on_categorical_slice(
            data=test_data,
            column_name=col,
            slice_value=slicevalue,
            categorical_features=cat_features,
            label=label,
            encoder=encoder,
            lb=lb,
            model=model,)

        with open("slice_output.txt", "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision:{p:.4f} | Recall:{r:.4f} | F1:{fb:.4f}", file=f)
