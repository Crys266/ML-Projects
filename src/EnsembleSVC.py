import pandas as pd
import json
import os
import numpy as np
from tqdm import tqdm
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

# Preparing feature vectors
print("Extracting features from JSON files...")
feature_list = []
labels = []
timestamps = []
all_features = set()

for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
    sha256 = row['sha256']
    label = row['label']
    json_file_path = os.path.join(json_directory, f"{sha256}.json")
    timestamp = row['timestamp']  # Assuming there is a 'timestamp' column in the dataset

    with open(json_file_path, 'r') as file:
        json_data = json.load(file)

    features = extract_features(json_data)
    all_features.update(features.keys())

    feature_list.append(features)
    labels.append(label)
    timestamps.append(timestamp)

print("Creating features DataFrame... ")
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

# Convert timestamps to datetime objects
timestamps = pd.to_datetime(timestamps)

# Splitting the dataset into training and test sets while maintaining temporal distribution
train_size = int(len(labels) * 0.85)
print('Train size: ', train_size)
X_train, X_test = sparse_matrix[:train_size], sparse_matrix[train_size:]
y_train, y_test = labels[:train_size], labels[train_size:]
timestamps_train = timestamps[:train_size]

# Extracting positive and negative samples from the training set
positive_samples = np.where(y_train == 1)[0]
print('Number of positive train samples: ', positive_samples.size)
negative_samples = np.where(y_train == 0)[0]
print('Number of negative train samples: ', negative_samples.size)

# Creating temporal bins (in this example we use 20 bins)
num_bins = 20
bins = pd.cut(timestamps_train, bins=num_bins, labels=False)

# Ensuring negative samples are evenly distributed across temporal bins
negative_bins = bins[negative_samples]
bin_indices = [np.where(negative_bins == i)[0] for i in range(num_bins)]

# Desired number of subsets
# The number of subsets is calculated considering the ratio of positive to negative samples to achieve balanced datasets
num_subsets = 9
subset_size = len(negative_samples) // num_subsets

# Preparing negative subsets
negative_subsets = [[] for _ in range(num_subsets)]

for bin_index in bin_indices:
    bin_data = negative_samples[bin_index]
    bin_data_shuffled = np.random.permutation(bin_data)
    for i in range(num_subsets):
        start_index = i * subset_size // num_bins
        end_index = (i + 1) * subset_size // num_bins
        negative_subsets[i].extend(bin_data_shuffled[start_index:end_index])

# Converting lists of negative subsets to numpy arrays
negative_subsets = [np.array(subset) for subset in negative_subsets]

# Creating balanced training datasets
training_subsets = []
for neg_subset in negative_subsets:
    indices = np.concatenate((positive_samples, neg_subset))
    np.random.shuffle(indices)  # Shuffle to mix positive and negative samples
    X_subset = X_train[indices]
    y_subset = y_train[indices]
    training_subsets.append((X_subset, y_subset))

# After multiple runs, the grid search almost always returns 1 for the C parameter
# to save execution time, we fix the parameter
# Train multiple SVC classifiers on each subset
classifiers = []
clf = SVC(kernel='linear', C=1, class_weight={0: 1, 1: 2}, probability=True)

print("Training Classifiers... ")
for X_subset, y_subset in training_subsets:
    clf.fit(X_subset, y_subset)
    classifiers.append(clf)

# Computing predictions on the ensemble of classifiers
ensemble_predictions = np.zeros((len(y_test), len(classifiers)))
print("Computing predictions...")

for i, classifier in enumerate(classifiers):
    y_score = classifier.decision_function(X_test)
    y_pred = (y_score > 0.0).astype(int)  # Use threshold to limit the number of false negatives, as it is
    ensemble_predictions[:, i] = y_pred   # assumed that misclassification of malware is the worst case

# Final voting (majority voting)
final_predictions = np.mean(ensemble_predictions, axis=1)
final_predictions = (final_predictions > 0.5).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, final_predictions)
recall = recall_score(y_test, final_predictions)
f1 = f1_score(y_test, final_predictions)
cm = confusion_matrix(y_test, final_predictions)

print(f'Accuracy: {accuracy}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print('Confusion Matrix\n', cm)

# Calculate ROC curve and AUC
y_score = np.mean([classifier.decision_function(X_test) for classifier in classifiers], axis=0)
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
results = []
results.append({
    'accuracy': accuracy,
    'f1_score': f1,
    'recall': recall,
    'confusion_matrix': cm,
    'fpr': fpr,
    'tpr': tpr,
    'roc_auc': roc_auc,
})

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Save all results to a file
with open('resultsEns.pkl', 'wb') as f:
    pickle.dump(results, f)

# Optionally, save results to a CSV for easy viewing
results_df = pd.DataFrame([{
    'accuracy': r['accuracy'],
    'f1_score': r['f1_score'],
    'recall': r['recall'],
    'confusion_matrix': r['confusion_matrix'],
    'roc_auc': r['roc_auc'],
} for r in results])

results_df.to_csv('resultsEns.csv', index=False)

print("Results saved.")
