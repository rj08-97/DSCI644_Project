import pandas as pd

from sklearn import preprocessing

from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from sklearn.metrics import classification_report

from project.my_evaluation import my_evaluation

from sklearn.feature_extraction.text import TfidfVectorizer


class my_model:
    def __init__(self):
        # defines the self function used in fit and predict
        self.preprocessor = TfidfVectorizer()
        self.clf = PassiveAggressiveClassifier(class_weight="balanced")

    def fit(self, X, y):
        X_num = X[["Del Func", "New Func", "Reached Del Func >= X"]]
        preprocess_num = preprocessing.normalize(X_num)
        scale_data = preprocessing.scale(preprocess_num)
        final_data_frame = pd.DataFrame(scale_data, columns=["Del Func", "New Func", "Reached Del Func >= X"])
        self.clf.fit(final_data_frame, y)
        return

    def predict(self, X):
        # remember to apply the same preprocessing in fit() on test data before making predictions
        X_num = X[["Del Func", "New Func", "Reached Del Func >= X"]]
        preprocess_num = preprocessing.normalize(X_num)
        scale_data = preprocessing.scale(preprocess_num)
        final_data_frame = pd.DataFrame(scale_data, columns=["Del Func", "New Func", "Reached Del Func >= X"])
        predictionsOfModel = self.clf.predict(final_data_frame)
        return predictionsOfModel


def test(data):
    y = data["Hit/Dismiss"]
    X = data.drop(['Hit/Dismiss'], axis=1)
    split_point = int(0.8 * len(y))
    X_train = X.iloc[:split_point]
    X_test = X.iloc[split_point:]
    y_train = y.iloc[:split_point]
    y_test = y.iloc[split_point:]
    clf = my_model()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    eval = my_evaluation(predictions, y_test)
    f1 = eval.f1(target=2)
    print(classification_report(y_test, predictions, target_names=["0", "1"]))
    return f1


if __name__ == "__main__":
    # Load data
    data = pd.read_csv("/Users/rakshandajha/PycharmProjects/DSCI644Project/data/Project4.csv")
    # Replace missing values with empty strings
    data = data.fillna("")
    f1 = test(data)
    print("F1 score: %f" % f1)

