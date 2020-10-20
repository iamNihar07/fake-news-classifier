import numpy
import pandas
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

#Read the data
dataset=pandas.read_csv('dataset\\news.csv')

#Display the nummber of rows and columns of the dataset
print(dataset.shape)

#Preview the first five tuples of the dataset
print(dataset.head())
