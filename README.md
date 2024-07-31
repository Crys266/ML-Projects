# ML-Projects: Android Malware Detection

This project is implemented using the training dataset provided by the [ELSA challenge](https://benchmarks.elsa-ai.eu/?ch=6&com=downloads).
This dataset is composed of 75K applications sampled between 2017-01-01 and 2019-12-31, with 25K applications per year and a proportion of 9:1 between goodware and malware samples.

Considering the strong imbalance of the dataset between positive and negative samples, two solutions have been proposed with the aim of limiting and reducing the number of false negatives, considering the incorrect classification of a malware as a legitimate sample as the worst case. 
In both solutions, in order to obtain concrete results, it is important to consider a correct temporal distribution of the samples between training and testing, so as not to obtain future samples in the training and vice versa.

The first proposed solution is implemented using an SVC classifier. In this case, we first proceed by balancing the classes using weights with the 'class_weight' parameter of the classifier. Although this modification results in a very high accuracy, the number of FNs remains high. Once the model is trained, we try to improve this factor by acting on the classifier's decision threshold to improve this characteristic. In this solution a consistent time division between train and test is performed via the TimeSeriesSplit method.

The second solution attempts to further minimize the number of FNs, but this results in a decrease in overall accuracy. In this implementation, once a portion of the test samples has been extracted, 9 new subsets are created, composed of the same number of negative and positive samples (the positives are the same for each dataset while the negatives change). Subsequently, 9 SVCs are trained, each on a different dataset, and finally an Ensemble is created to perform the predictions.
