import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Read the data
dataset=pandas.read_csv('dataset\\news.csv')

# Get the number of rows and columns of the dataset
x,y = dataset.shape
# print(dataset.shape)

# #Preview the first five tuples of the dataset
# print(dataset.head())

# print()
# get the labels of REAL/FAKE from the dataset
label = dataset.label
text = dataset.text
# #Preview the first five labels of the dataset
# print(label.head())

# Split the dataset into training and testing sets
# Splitting the dataset into 20% testing and 80% training
# Splitting is always done with the same random int 42, so output is reproducible across multiple function calls
TrainX, TestX, TrainY, TestY = train_test_split(text, label, test_size=0.2, train_size=0.8, random_state=42)

# print(TrainX)
# print()
# print(TestX)
# print()
# print(TrainY)
# print()
# print(TestY)

# Initialize TfidfVectorizer with stop words from the English language and a maximum document frequency of 0.7
TFIDFVect=TfidfVectorizer(stop_words='english', max_df=0.7)

# Learn vocabulary and idf, return document-term matrix. i.e. fit and transform the vectorizer on the train set
TFIDFTrain=TFIDFVect.fit_transform(TrainX) 
# Transform documents to document-term matrix i.e transform the vectorizer on the test set
# Uses the vocabulary and document frequencies (df) learned by fit_transform
TFIDFTest=TFIDFVect.transform(TestX)

# print(TFIDFTrain)
# print()
# print(TFIDFTest)

# Initialize a PassiveAggressiveClassifier, with maximum number of passes over train data to be 50
classifier=PassiveAggressiveClassifier(max_iter=50)
# Fit linear model of (x,y) = (TFIDTrain, TrainY) with Passive Aggressive algorithm
classifier.fit(TFIDFTrain,TrainY)

# Predict class labels for samples in TFIDFTest
PredY = classifier.predict(TFIDFTest)
# Calculate the accuracy score of the classification of (TestY, PredY) = (correct labels, predicted labels by the classifier)
score = accuracy_score(TestY,PredY)
score = round(score*100, 2)
print()
print("Model Classifier Accuracy: %0.2f%%" %(score))

# Compute confusion matrix to evaluate the accuracy of a classification
confMatrix = confusion_matrix(TestY,PredY, labels=['FAKE','REAL'])
print("DATASET INSIGHTS - For the dataset of total %d news articles: " %(x))
print("After training the model using %d news articles, " %(0.8*x))
print("Out of %d tested news articles, we have: " %(0.2*x))
print("Number of articles which are FAKE, and correctly classified as FAKE  : %d" %(confMatrix[0][0]))
print("Number of articles which are FAKE, and incorrectly classified as REAL: %d" %(confMatrix[0][1]))
print("Number of articles which are REAL, and incorrectly classified as FAKE: %d" %(confMatrix[1][0]))
print("Number of articles which are REAL, and correctly classified as REAL  : %d" %(confMatrix[1][1]))
print()
