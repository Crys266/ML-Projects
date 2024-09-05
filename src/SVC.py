import pandas as pd
import json
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_curve, auc
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from utils import extract_features
import pickle


# Loading the dataset
csv_file_path = "training_set.csv"
print(f"Loading dataset from {csv_file_path}...")
dataset = pd.read_csv(csv_file_path)

# Directory of JSON files
json_directory = "training_set_features"

# Preparing the feature vectors
print("Extracting features from JSON files...")
feature_list = []
labels = []
all_features = set()

for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
    sha256 = row['sha256']
    label = row['label']
    json_file_path = os.path.join(json_directory, f"{sha256}.json")

    with open(json_file_path, 'r') as file:
        json_data = json.load(file)

    features = extract_features(json_data)
    all_features.update(features.keys())

    feature_list.append(features)
    labels.append(label)

print("Creating features DataFrame...")
feature_index = {feature: i for i, feature in enumerate(all_features)}

data = []
rows = []
cols = []

for row_index, features in enumerate(feature_list):
    for feature, value in features.items():
        rows.append(row_index)
        cols.append(feature_index[feature])
        data.append(value)

sparse_matrix = csr_matrix((data, (rows, cols)), shape=(len(feature_list), len(all_features)))
labels = np.array(labels)

# Create the TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
print('tscv: ', tscv)

# Iterate through the TimeSeriesSplit splits and analyze class distribution
for fold, (train_index, test_index) in enumerate(tscv.split(sparse_matrix)):
    y_train, y_test = labels[train_index], labels[test_index]

# Iterate through the TimeSeriesSplit splits
accuracies = []
results = []
i = 0
model = SVC(kernel='linear', class_weight='balanced', probability=True)

for fold, (train_index, test_index) in enumerate(tscv.split(sparse_matrix)):
    # Split data into training and test sets using the indices generated by TimeSeriesSplit
    X_train, X_test = sparse_matrix[train_index], sparse_matrix[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # Print the size of training and test samples
    print(f"Fold {fold + 1}")
    print(f"  Number of training samples: {len(train_index)}")
    print(f"  Number of test samples: {len(test_index)}")

    # Train the linear SVM model
    print("  Training SVC model...")
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_score = model.decision_function(X_test)
    y_pred = (y_score > 0.0).astype(int)  # Change the decision threshold (default 0.0)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f'  Accuracy: {accuracy}')

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print('   Confusion Matrix\n', cm)

    # Compute recall score
    recall = recall_score(y_test, y_pred)
    print('   Recall:', recall)

    # Compute f1 score
    f1 = f1_score(y_test, y_pred)
    print('   F1 Score:', f1)

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    results.append({
        'subset': i + 1,
        'accuracy': accuracy,
        'f1_score': f1,
        'recall': recall,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
    })
    i += 1

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('SVC Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


# Save all results to a file
with open('resultsSVC.pkl', 'wb') as f:
    pickle.dump(results, f)

# Optionally, save results to a CSV for easy viewing
results_df = pd.DataFrame([{
    'subset': r['subset'],
    'accuracy': r['accuracy'],
    'f1_score': r['f1_score'],
    'recall': r['recall'],
    'confusion_matrix': r['confusion_matrix'],
    'roc_auc': r['roc_auc'],
} for r in results])

results_df.to_csv('resultsSVC.csv', index=False)

print("Results saved.")
