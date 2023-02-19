# Sampling_Assignment

The act of selecting a portion of data from a larger dataset is referred to as sampling. In statistics and machine learning, sampling is frequently used to balance unbalanced datasets or to produce training and testing sets. There are numerous sampling methods available, such as:

1. Simple random sampling: unbiasedly choosing data points at random from the dataset.


2. Stratified sampling: To ensure that the sample is representative of the population, the dataset is divided into subgroups (or strata), and samples are taken from each subgroup.

3. Systematic sampling: In systematic sampling, a sample is created by choosing every "kth" person from a population of "N" people, where "k" is a constant that depends on the population's overall size as well as the required sample size.

4. Cluster sampling: Using a natural grouping factor, cluster sampling divides the population into clusters (or groups) (such as geographic location or occupation).

STEPS FOR ACCOMPLISHING THE TASKS ARE:-

The First step is to download the dataset from the given link : “https://github.com/AnjulaMehto/Sampling\_Assignment/blob/main/Creditcard\_data.csv “

We can use one of the strategies to balance the dataset because the maximum value of the class in the given dataset makes it unbalanced. To balance the dataset, for instance, we can apply an oversampling technique like Synthetic Minority Over-sampling Technique (SMOTE) or Adaptive Synthetic Sampling (ADASYN). To carry out these strategies, we can use libraries like imbalanced-learn or SMOTE-variants.

We will calculate the sample size for five samples using the sample size detection formula. The samples can then be created using random sampling approaches as simple random sampling, stratified sampling, or systematic sampling.

Depending on the problem and the data, we can select one of five sampling strategies, including simple random sampling, stratified sampling, systematic sampling, cluster sampling, and multistage sampling. Next, using five distinct ML models—logistic regression, decision trees, random forests, support vector machines, or neural networks—we can use these strategies. We can make use of tools like scikit-learn or TensorFlow to put these strategies into practise.

The data can be divided into training and testing sets, the models can be fitted on the training set, and the performance on the testing set can be assessed using the relevant metrics, such as accuracy, precision, recall, F1-score, or ROC-AUC.

We can change the code by keeping note of the maximum accuracy and the corresponding sampling technique for each model in order to figure out which sampling technique delivers a higher accuracy for each model.
