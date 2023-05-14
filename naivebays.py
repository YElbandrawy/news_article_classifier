import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# load dataset
data = pd.read_csv('uci-news-aggregator.csv')

# split dataset into training and testing sets
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# create document-term matrix using bag of words approach
vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(train_data['TITLE'])
X_test = vectorizer.transform(test_data['TITLE'])
y_train = train_data['CATEGORY']
y_test = test_data['CATEGORY']

# instantiate the classifier and fit the model to the training data
clf = MultinomialNB()
clf.fit(X_train, y_train)

# make predictions on the testing data
y_pred = clf.predict(X_test)

# print accuracy score
print("Accuracy: ", accuracy_score(y_test, y_pred))
