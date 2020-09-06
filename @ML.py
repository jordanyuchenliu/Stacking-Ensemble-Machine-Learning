import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

import numpy as np
import re

if __name__ == '__main__':
    # separate the data into non-numeric (item title) and numeric (every other features)
    data = pd.read_excel('/Users/jordanliu/Desktop/google_drive/amazon/zfinish/Final_Results_wo30.xlsx', index_col = 0)
    data = data.reset_index(drop=True)
    word_data = data[['Item Title']]
    num_data = data.drop(['Item Title', 'SKU', 'Target'], axis = 'columns')
    target = data['Target']

    # NON-NUMERIC SECTION
    # remove all unnecessary spaces, numbers or any string containing numbers (such as "10086a") in "Item Title" column
    for x in range(len(word_data)):
        if any(str.isdigit(c) for c in word_data.at[x,'Item Title']):
            word_data.at[x,'Item Title'] = re.sub('\w*\d\w*', '', word_data.at[x,'Item Title'])
        word_data.at[x, 'Item Title'] = ' '.join(word_data.at[x, 'Item Title'].split())

    # run the non-numeric data using Naive Bayes model
    # First we need to separate each unique word in different columns
    text_x_train, text_x_test, text_y_train, text_y_test = train_test_split(word_data['Item Title'], target, test_size = 0.5)
    v = CountVectorizer()
    x_train_col = v.fit_transform(text_x_train.values)
    x_test_col = v.transform(text_x_test.values)

    multiNB_model = MultinomialNB()
    multiNB_model.fit(x_train_col, text_y_train)
    print(multiNB_model.score(x_test_col, text_y_test))

    # because v.get_feature_names() is in method object type and .coef is in a 2-D array, we need to convert them into lists
    # After conversion, using zip and sort to find the top 10 keywords with highest probability
    feature_name_list = [elem for elem in v.get_feature_names()]
    log_prob_list = [elem for elem in multiNB_model.coef_[0]]
    feature_prob_zip = list(zip(feature_name_list, log_prob_list))
    feature_prob_zip = sorted(feature_prob_zip, key = lambda x:x[1], reverse = True)  # this is sorting zip list based on second elem in tuple
    print(feature_prob_zip[:10])


    # create a column of the predicted probabilities of x_train and x_test in word_data to later combine with the
    # numeric (dense) features

    x_train_index = text_x_train.index.tolist()
    x_test_index = text_x_test.index.tolist()
    x_train_predict_prob = [elem[1] for elem in multiNB_model.predict_proba(x_train_col)]
    x_test_predict_prob = [elem[1] for elem in multiNB_model.predict_proba(x_test_col)]
    x_index_predict_zip = list(zip(x_test_index + x_train_index, x_test_predict_prob + x_train_predict_prob))
    sorted_index, sorted_predict_prob = zip(*sorted(x_index_predict_zip))       # this is sorting zip list based on first elem in tuple

    print(sorted_predict_prob[:10])
    num_data.insert(0, 'Text Predict Prob', sorted_predict_prob)

    # NUMERIC SECTION
    # need to first convert Primary Category & Second Category into numbers (LabelEncoder)
    # notice CountVectorizer separates every unique word inside a string into different columns with value of 0 or 1
    # LabelEncoder is labeling every unique string inside a column with number
    le = LabelEncoder()
    num_data['Primary Category'] = le.fit_transform(num_data['Primary Category'])
    primary_cat_le_num = list(np.unique(num_data['Primary Category']))
    primary_cat_le_text = list(le.inverse_transform(primary_cat_le_num))
    primary_cat_le_zip = dict(zip(primary_cat_le_num, primary_cat_le_text))
    print(primary_cat_le_zip)

    num_data['Secondary Category'] = le.fit_transform(num_data['Secondary Category'])
    sec_cat_le_num = list(np.unique(num_data['Secondary Category']))
    sec_cat_le_text = list(le.inverse_transform(sec_cat_le_num))
    sec_cat_le_zip = dict(zip(sec_cat_le_num, sec_cat_le_text))         # zip is created with Primary & Secondary as reference for the LabelEncoder()

    # run the numeric data with Linear kernel SVC
    # need to first scale the columns within the range [0,1] to optimize the speed of ML execution
    minmax_scaler = MinMaxScaler()
    num_data[num_data.columns] = minmax_scaler.fit_transform(num_data[num_data.columns])

    # GridSearchCV essentially performs KFold validation in every variation of 'C' and 'gamma' as specified to find
    # the most optimal parameter with highest accuracy
    """""
    from sklearn.model_selection import GridSearchCV
    clf = GridSearchCV(SVC(kernel = 'linear'), {
        'C': [0.01, 0.1, 1, 10, 20, 30],
        'gamma' : [0.01, 0.1, 1, 10,100]
    },cv = 5, return_train_score=False
    )

    clf.fit(num_data, target)
    print(clf.best_score_, clf.best_params_)
    """""
    # now we know when 'C' = 30, 'gamma = 0.01 from clf.best_score_ & clf.best_params_, we need to use KFold Cross
    # Validation to calculate the log prob of features in each iteration and take the average
    num_data = num_data[['Amazon Price', 'Amazon Review', 'Amazon Ratings']]

    from sklearn.model_selection import StratifiedKFold
    folds = StratifiedKFold(n_splits = 3)
    svc_score = []
    svc_coef = []
    svc_coef_mean = []
    for train_index, test_index in folds.split(num_data, target):
        print(train_index, test_index)
        num_x_train = num_data.iloc[train_index]
        num_x_test = num_data.iloc[test_index]
        num_y_train = target.iloc[train_index]
        num_y_test = target.iloc[test_index]

        svc_model = SVC(C=30, gamma = 0.01, kernel = 'linear')
        svc_model.fit(num_x_train, num_y_train)
        svc_score.append(svc_model.score(num_x_test, num_y_test))
        svc_coef.append(svc_model.coef_[0])
    print(svc_coef)
    svc_score = np.mean(svc_score)
    svc_coef_mean = [(svc_coef[0][x] + svc_coef[1][x] + svc_coef[2][x])/3
                     for x in range(len(svc_coef[0]))]
    print(svc_score)
    print(svc_coef_mean)

    num_data_feature_logprob_zip = (list(zip(num_data.columns.values.tolist(), svc_coef_mean)))
    num_data_feature_logprob_zip = sorted(num_data_feature_logprob_zip, key = lambda x: abs(x[1]), reverse = True)
    print(num_data_feature_logprob_zip)






