import pandas as pd
import json
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score,roc_curve, auc
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from utils import extract_features


# Caricamento del dataset
csv_file_path = "training_set.csv"
print(f"Loading dataset from {csv_file_path}...")
dataset = pd.read_csv(csv_file_path)

# Directory dei file JSON
json_directory = "training_set_features"

# Preparazione dei feature vectors
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

# Convertiamo i timestamp in oggetti datetime
timestamps = pd.to_datetime(timestamps)

# Suddividere il dataset in training e test set mantenendo la distribuzione temporale
train_size = int(len(labels) * 0.85)
print('Train size: ', train_size)
X_train, X_test = sparse_matrix[:train_size], sparse_matrix[train_size:]
y_train, y_test = labels[:train_size], labels[train_size:]
timestamps_train = timestamps[:train_size]

# Estraiamo i campioni positivi e negativi dal training set
positive_samples = np.where(y_train == 1)[0]
print('Number of positive train samples: ', positive_samples.size)
negative_samples = np.where(y_train == 0)[0]
print('Number of negative train samples: ', negative_samples.size)

# Creiamo bin temporali (in questo esempio usiamo 20 bin)
num_bins = 20
bins = pd.cut(timestamps_train, bins=num_bins, labels=False)

# Assicuriamoci che i campioni negativi siano distribuiti uniformemente nei bin temporali
negative_bins = bins[negative_samples]
bin_indices = [np.where(negative_bins == i)[0] for i in range(num_bins)]

# Numero di sottoinsiemi desiderato
num_subsets = 9
subset_size = len(negative_samples) // num_subsets

# Preparazione dei sottoinsiemi negativi
negative_subsets = [[] for _ in range(num_subsets)]

for bin_index in bin_indices:
    bin_data = negative_samples[bin_index]
    bin_data_shuffled = np.random.permutation(bin_data)
    for i in range(num_subsets):
        start_index = i * subset_size // num_bins
        end_index = (i + 1) * subset_size // num_bins
        negative_subsets[i].extend(bin_data_shuffled[start_index:end_index])

# Convertiamo le liste di sottoinsiemi negativi in array numpy
negative_subsets = [np.array(subset) for subset in negative_subsets]

# Creazione dei dataset di training bilanciati
training_subsets = []
for neg_subset in negative_subsets:
    indices = np.concatenate((positive_samples, neg_subset))
    np.random.shuffle(indices)  # Shuffle per mescolare positivi e negativi
    X_subset = X_train[indices]
    y_subset = y_train[indices]
    training_subsets.append((X_subset, y_subset))

# Addestrare più classificatori SVC su ogni subset
classifiers = []
param_grid = {'C': [1, 10]}
grid_search = GridSearchCV(SVC(kernel='linear', class_weight='balanced', probability=True), param_grid, scoring='accuracy', cv=3, verbose=1, n_jobs=-1)

for X_subset, y_subset in training_subsets:
    grid_search.fit(X_subset, y_subset)
    best_model = grid_search.best_estimator_
    classifiers.append(best_model)
    print(f"Best parameters found: {grid_search.best_params_}")

# Fare predizioni sull'ensemble di classificatori
ensemble_predictions = np.zeros((len(y_test), len(classifiers)))
print("Computing predictions...")

for i, classifier in enumerate(classifiers):
    y_score = classifier.decision_function(X_test)
    y_pred = (y_score > -0.5).astype(int)  # Usa la soglia standard
    ensemble_predictions[:, i] = y_pred

# Votazione finale (majority voting)
final_predictions = np.mean(ensemble_predictions, axis=1)
final_predictions = (final_predictions > 0.5).astype(int)

# Valutare il modello
accuracy = accuracy_score(y_test, final_predictions)
recall = recall_score(y_test, final_predictions)
f1 = f1_score(y_test, final_predictions)
cm = confusion_matrix(y_test, final_predictions)

print(f'Accuracy: {accuracy}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print('Confusion Matrix\n', cm)

# Calcolare la curva ROC e l'AUC
y_score = np.mean([classifier.decision_function(X_test) for classifier in classifiers], axis=0)
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# Stampare la curva ROC
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



















"""
# Creare il TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
print('tscv: ', tscv)

# Iterare attraverso le divisioni del TimeSeriesSplit e analizzare la distribuzione delle classi
for fold, (train_index, test_index) in enumerate(tscv.split(sparse_matrix)):
    y_train, y_test = labels[train_index], labels[test_index]
    print(f"Fold {fold + 1}:")
    print(f"  Label Training set: {np.bincount(y_train)}")
    print(f"  Label Test set: {np.bincount(y_test)}")

# Parametri per la ricerca a griglia
param_grid = {
    'C': [1, 10, 100],  # Parametro di regolarizzazione
}

# Eseguire GridSearchCV su una delle divisioni di TimeSeriesSplit
train_index, val_index = next(tscv.split(sparse_matrix))
X_train, X_val = sparse_matrix[train_index], sparse_matrix[val_index]
y_train, y_val = labels[train_index], labels[val_index]

# Eseguire GridSearchCV con SVC lineare
grid_search = GridSearchCV(SVC(kernel='linear', class_weight='balanced', probability=True), param_grid, scoring='accuracy', cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Migliori parametri trovati
print(f"Best parameters found: {grid_search.best_params_}")


# Iterare attraverso le divisioni del TimeSeriesSplit
accuracies = []
model = grid_search.best_estimator_

# Migliori parametri trovati
# print(f"Best parameters found: {grid_search.best_params_}")
for fold, (train_index, test_index) in enumerate(tscv.split(sparse_matrix)):
    # Divisione dei dati in training e test set usando gli indici generati da TimeSeriesSplit
    X_train, X_test = sparse_matrix[train_index], sparse_matrix[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # Stampare la dimensione dei campioni di training e test
    print(f"Fold {fold + 1}")
    print(f"  Number of training samples: {len(train_index)}")
    print(f"  Number of test samples: {len(test_index)}")

    # Addestrare il modello SVM lineare
    print("  Training model...")
    model.fit(X_train, y_train)

    # Predire e valutare
    # y_pred = model.predict(X_test)
    y_score = model.decision_function(X_test)
    y_pred = (y_score > -0.8).astype(int)  # Cambia la soglia di decisione (default 0.0)
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

    # Calcolare la curva ROC e l'AUC
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # Stampare la curva ROC
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

# Calcolare l'accuratezza media
mean_accuracy = sum(accuracies) / len(accuracies)
print('Mean Accuracy:', mean_accuracy)
"""""