import time
import sys
import pandas as pd
from project.experiment_DecisionTree import my_model
sys.path.insert(0, '../..')
from project.my_evaluation import my_evaluation

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
    f1 = eval.f1(target=1)
    return f1


if __name__ == "__main__":
    start = time.time()
    # Load data
    data = pd.read_csv("/Users/rakshandajha/PycharmProjects/DSCI644Project/data/Project4.csv")
    # Replace missing values with empty strings
    data = data.fillna("")
    f1 = test(data)
    print("F1 score: %f" % f1)
    runtime = (time.time() - start) / 60.0
    print(runtime)