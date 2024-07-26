# Importing the libraries
import pandas as pd
import json
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC  # Importing LinearSVC instead of SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.sparse import csr_matrix

# Funzione per estrarre le caratteristiche dai file JSON secondo la struttura DREBIN
def extract_features(json_data):
    feature_types = {
        'features': 'S1_',
        'req_permissions': 'S2_',
        'activities': 'S3_',
        'services': 'S3_',
        'providers': 'S3_',
        'receivers': 'S3_',
        'intent_filters': 'S4_',
        'api_calls': 'S5_',
        'used_permissions': 'S6_',
        'suspicious_calls': 'S7_',
        'urls': 'S8_'
    }

    features = {}

    # Inizializzazione delle caratteristiche con valori binari (0 o 1)
    for feature_type, prefix in feature_types.items():
        if feature_type in json_data:
            for item in json_data[feature_type]:
                features[f'{prefix}{item}'] = 1

    return features

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
all_features = set()

for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
    sha256 = row['sha256']
    label = row['label']
    json_file_path = os.path.join(json_directory, f"{sha256}.json")

    # Caricamento del file JSON
    with open(json_file_path, 'r') as file:
        json_data = json.load(file)

    # Estrazione delle caratteristiche
    features = extract_features(json_data)
    all_features.update(features.keys())

    # Aggiunta delle caratteristiche e della label alla lista
    feature_list.append(features)
    labels.append(label)

print("Creating features DataFrame...")
# Creazione di un mapping delle caratteristiche a indici
feature_index = {feature: i for i, feature in enumerate(all_features)}

# Creazione delle matrici sparse
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

# Suddivisione del dataset in training set e test set evitando data snooping
print("Splitting dataset into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(sparse_matrix, labels, test_size=0.3, random_state=42, stratify=labels)

# Feature Scaling
print("Scaling features...")
sc = StandardScaler(with_mean=False)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Inizializzazione del classificatore LinearSVC
classifier = LinearSVC(verbose=1, max_iter=20000, C=0.1)

# Addestramento del modello
print("Training LinearSVC...")
classifier.fit(X_train, y_train)

# Predicting the Test set results
print("Predicting test set results...")
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix\n', cm)

# Compute accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Compute precision score
precision = precision_score(y_test, y_pred, average='micro')
print('Precision:', precision)

# Compute recall score
recall = recall_score(y_test, y_pred)
print('Recall:', recall)

# Compute f1 score
f1 = f1_score(y_test, y_pred)
print('F1 Score:', f1)
