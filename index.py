import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.model_selection import train_test_split


def load_dataset(dataset):
	# get dataset as np array
	data = np.array(pd.read_csv(dataset, delimiter=',', usecols=['Rating', 'Reviews'],  engine='c', compression='gzip', encoding='utf-8'))
	# delete invalid data
	delete_indexes = []
	for i in range(len(data)):
		if not isinstance(data[i][0], float) or not isinstance(data[i][1], str):
			delete_indexes.append(i)
	data = np.delete(data, delete_indexes, axis=0)

	text = data[:, 1]
	# get rating as label
	label_1 = data[:, 0]
	# 5 class Sentiment labels to -2, -1, 0, 1, 2
	label_1 = label_1 - 3
	# Split train and test
	text_train, text_test, label_1_train, label_1_test = train_test_split(text, label_1, test_size=0.2, random_state=42)
	# 3 class Sentiment labels to -1, 0, 1,
	label_2_train = np.array([-1. if rating < 0 else 1. if rating > 0 else 0. for rating in label_1_train])
	label_2_test = np.array([-1. if rating < 0 else 1. if rating > 0 else 0. for rating in label_1_test])

	return text_train, label_1_train, label_2_train, text_test, label_1_test, label_2_test


# 413840 amazon review
DATASET = 'C:/Users/farsh/Desktop/پایان نامه/شبیه سازی/Sentiment Dataset/English/Amazon Unlocked Mobile/Amazon_Unlocked_Mobile.csv.gz'
POLARITY_NEUTRAL_DOMAIN = 0.2
text_train, label_1_train, label_2_train, text_test, label_1_test, label_2_test = load_dataset(DATASET)

# Predict label 2 for train data using text blob
tb_label2_predict = []
for i in range(len(text_train)):
	polarity = TextBlob(text_train[i]).sentiment.polarity
	if polarity > POLARITY_NEUTRAL_DOMAIN:
		predict = 1
	elif polarity < -1 * POLARITY_NEUTRAL_DOMAIN:
		predict = -1
	else:
		predict = 0
	tb_label2_predict.append(predict)

tb_label2_predict = np.array(tb_label2_predict)
# Compare

compare = (tb_label2_predict == label_2_train)

correct = 0
for i in compare:
	if i == True:
		correct = correct + 1
print(correct/len(compare))
