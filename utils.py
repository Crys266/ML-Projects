

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

    for feature_type, prefix in feature_types.items():
        if feature_type in json_data:
            for item in json_data[feature_type]:
                features[f'{prefix}{item}'] = 1

    return features
