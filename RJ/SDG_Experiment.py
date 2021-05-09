import pandas as pd

from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from project.my_evaluation import my_evaluation

import seaborn as sns
import numpy as np



class my_model:
    def __init__(self):
        # defines the self function used in fit and predict
        self.clf = SGDClassifier(class_weight="balanced", n_iter_no_change=5)

    def fit(self, X, y):
        X_num = pd.DataFrame(X).drop(['Commit A'], axis=1).drop(['Commit B'], axis=1).drop(['Benchmark'], axis=1).drop(
            ['Top Chg by Instr >= X%'], axis=1).drop(['AltAvgLineBlank'], axis=1)
        X_norm = MinMaxScaler().fit_transform(X_num)
        chi_selector = SelectKBest(chi2, k=8)
        chi_selector.fit(X_norm, y)
        chi_support = chi_selector.get_support()
        chi_feature = X_num.loc[:, chi_support].columns.tolist()
        print(chi_feature)
        selected_feature_data = X_num[
            ['New Func >= X', 'Reached Del Func >= X', 'Top Chg by Call >= X%',
             'Top > X% by Call Chg by >= 10%', 'Top Chg Len >= X%', 'AltCountLineBlank', 'AltCountLineCode',
             'AltCountLineComment',
             'CountLine', 'CountLineBlank', 'CountLineCode', 'CountLineCodeDecl', 'CountLineCodeExe',
             'CountLineComment',
             'CountLineInactive', 'CountLinePreprocessor', 'CountSemicolon', 'CountStmt', 'CountStmtDecl',
             'CountStmtEmpty',
             'CountStmtExe', 'SumCyclomatic', 'SumCyclomaticModified', 'SumCyclomaticStrict', 'SumEssential',
             'MaxCyclomatic',
             'MaxCyclomaticModified', 'MaxNesting', 'AltAvgLineCode', 'AltAvgLineComment', 'AvgEssential', 'AvgLine',
             'AvgLineBlank', 'AvgLineCode', 'AvgLineComment']]
        data_pre = MinMaxScaler().fit_transform(selected_feature_data)
        preprocess_num = preprocessing.normalize(selected_feature_data)
        scale_data = preprocessing.scale(preprocess_num)
        final_data_frame = pd.DataFrame(scale_data,
                                        columns=['New Func >= X', 'Reached Del Func >= X', 'Top Chg by Call >= X%',
                                                 'Top > X% by Call Chg by >= 10%', 'Top Chg Len >= X%',
                                                 'AltCountLineBlank', 'AltCountLineCode', 'AltCountLineComment',
                                                 'CountLine', 'CountLineBlank', 'CountLineCode', 'CountLineCodeDecl',
                                                 'CountLineCodeExe', 'CountLineComment',
                                                 'CountLineInactive', 'CountLinePreprocessor', 'CountSemicolon',
                                                 'CountStmt', 'CountStmtDecl', 'CountStmtEmpty',
                                                 'CountStmtExe', 'SumCyclomatic', 'SumCyclomaticModified',
                                                 'SumCyclomaticStrict', 'SumEssential', 'MaxCyclomatic',
                                                 'MaxCyclomaticModified', 'MaxNesting', 'AltAvgLineCode',
                                                 'AltAvgLineComment', 'AvgEssential', 'AvgLine',
                                                 'AvgLineBlank', 'AvgLineCode', 'AvgLineComment'])
        self.clf.fit(final_data_frame, y)
        return

    def predict(self, X):
        X_num = pd.DataFrame(X).drop(['Commit A'], axis=1).drop(['Commit B'], axis=1).drop(['Benchmark'], axis=1).drop(
            ['Top Chg by Instr >= X%'], axis=1).drop(['AltAvgLineBlank'], axis=1)
        selected_feature_data = X_num[
            ['New Func >= X', 'Reached Del Func >= X', 'Top Chg by Call >= X%',
             'Top > X% by Call Chg by >= 10%', 'Top Chg Len >= X%', 'AltCountLineBlank', 'AltCountLineCode',
             'AltCountLineComment',
             'CountLine', 'CountLineBlank', 'CountLineCode', 'CountLineCodeDecl', 'CountLineCodeExe',
             'CountLineComment',
             'CountLineInactive', 'CountLinePreprocessor', 'CountSemicolon', 'CountStmt', 'CountStmtDecl',
             'CountStmtEmpty',
             'CountStmtExe', 'SumCyclomatic', 'SumCyclomaticModified', 'SumCyclomaticStrict', 'SumEssential',
             'MaxCyclomatic',
             'MaxCyclomaticModified', 'MaxNesting', 'AltAvgLineCode', 'AltAvgLineComment', 'AvgEssential', 'AvgLine',
             'AvgLineBlank', 'AvgLineCode', 'AvgLineComment']]
        preprocess_num = preprocessing.normalize(selected_feature_data)
        scale_data = preprocessing.scale(preprocess_num)
        final_data_frame = pd.DataFrame(scale_data,
                                        columns=['New Func >= X', 'Reached Del Func >= X', 'Top Chg by Call >= X%',
                                                 'Top > X% by Call Chg by >= 10%', 'Top Chg Len >= X%',
                                                 'AltCountLineBlank', 'AltCountLineCode', 'AltCountLineComment',
                                                 'CountLine', 'CountLineBlank', 'CountLineCode', 'CountLineCodeDecl',
                                                 'CountLineCodeExe', 'CountLineComment',
                                                 'CountLineInactive', 'CountLinePreprocessor', 'CountSemicolon',
                                                 'CountStmt', 'CountStmtDecl', 'CountStmtEmpty',
                                                 'CountStmtExe', 'SumCyclomatic', 'SumCyclomaticModified',
                                                 'SumCyclomaticStrict', 'SumEssential', 'MaxCyclomatic',
                                                 'MaxCyclomaticModified', 'MaxNesting', 'AltAvgLineCode',
                                                 'AltAvgLineComment', 'AvgEssential', 'AvgLine',
                                                 'AvgLineBlank', 'AvgLineCode', 'AvgLineComment'])
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
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    print(classification_report(y_test, predictions, target_names=["0", "1"]))

    y_true = ["hit", "miss"]
    y_pred = ["hit", "miss"]
    df_cm = pd.DataFrame(cm, columns=np.unique(y_true), index=np.unique(y_true))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    sns.heatmap(df_cm, annot=True)
    return f1


if __name__ == "__main__":
    # Load data
    data = pd.read_csv("/Users/rakshandajha/PycharmProjects/DSCI644Project/data/updatedData.csv")
    # Replace missing values with empty strings
    data = data.fillna("")
    f1 = test(data)
    print("F1 score: %f" % f1)

