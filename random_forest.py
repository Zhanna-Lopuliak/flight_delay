import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import accuracy_score

def get_cleaned_data():
    data = pd.read_csv('unique/all_final2.csv')
    data[['STD', 'ATD', 'STA', 'ATA']] = data[['STD', 'ATD', 'STA', 'ATA']].apply(pd.to_datetime)
    data['month'] = [i.month for i in data.STD.to_list()]
    data['month_day'] = [i.day for i in data.STD.to_list()]

    data_for_rf = data.drop(
        columns=['STD', 'ATD', 'STA', 'ATA', 'origin_icao', 'destination_icao', 'Unnamed: 0', 'departure_delay',
                 'arrival_delay'])

    for feature in data_for_rf.columns:
        if data_for_rf[feature].dtype == 'object' or data_for_rf[feature].dtype == 'bool':
            data_for_rf[feature] = pd.Categorical(data_for_rf[feature]).codes

    return data_for_rf


def form_df_benchmarks(X_train, y_train, X_test, y_test, name):
    try:

        n_estimators, max_depth, max_features, min_samples_leaf, min_samples_split, times = [], [], [], [], [], []

        acc_train, acc_test, oob = [], [], []

        for n_estimators_ in [30, 100, 400]:
            for max_depth_ in [12, 20]:
                for max_features_ in [9, 18]:
                    for min_samples_leaf_ in [100, 400]:
                        for min_samples_split_ in [300, 1000]:
                            n_estimators.append(n_estimators_)
                            max_depth.append(max_depth_)
                            max_features.append(max_features_)
                            min_samples_leaf.append(min_samples_leaf_)
                            min_samples_split.append(min_samples_split_)
                            print('n_estimators, max_depth, max_features, min_samples_leaf, min_samples_split =',
                                  n_estimators_, max_depth_, max_features_, min_samples_leaf_, min_samples_split_)
                            start_time = time.time()
                            rfcl = RandomForestClassifier(n_estimators=n_estimators_, oob_score=True,
                                                          max_depth=max_depth_,
                                                          max_features=max_features_,
                                                          min_samples_leaf=min_samples_leaf_,
                                                          min_samples_split=min_samples_split_, bootstrap=True,
                                                          random_state=1)
                            rfcl.fit(X_train, y_train)
                            time_ = time.time() - start_time
                            print("--- %s seconds ---" % (time.time() - start_time))
                            times.append(round(time_, 1))
                            oob.append(rfcl.oob_score_)
                            train_score = accuracy_score(y_train, rfcl.predict(X_train))
                            acc_train.append(train_score)
                            test_score = accuracy_score(y_test, rfcl.predict(X_test))
                            acc_test.append(test_score)
                            print('TRAIN SCORE', train_score, 'TEST SCORE', test_score, '\n\n')
    except:
        print('oh wow why')

    try:
        df = pd.DataFrame(list(zip(acc_train, acc_test, oob, n_estimators, max_depth, max_features, min_samples_leaf,
                                   min_samples_split, times)),
                          columns=['acc_train', 'acc_test', 'oob_score', 'n_trees', 'max_depth', 'max_features',
                                   'min_samples_leaf', 'min_samples_split', 'times'])

        df.to_csv('other/random_forest/' + name + '_rf.csv', index=False)
    except:
        print('there was a problem sorry')


def iterate_over_dependent_to_csvs(data):
    dependent = ['is_delayed_15_departure', 'is_delayed_15_arrival',
                 'is_delayed_30_departure', 'is_delayed_30_arrival',
                 'is_delayed_60_departure', 'is_delayed_60_arrival',
                 'is_delayed_90_departure', 'is_delayed_90_arrival',
                 'is_delayed_120_departure', 'is_delayed_120_arrival',
                 'delayed_group_d', 'delayed_group_a', 'is_delayed_arrival', 'is_delayed_departure']
    X = data.drop(columns=dependent)

    s_t = ['is_delayed_30_departure', 'is_delayed_30_arrival',
           'is_delayed_60_departure', 'is_delayed_60_arrival',
           'is_delayed_90_departure', 'is_delayed_90_arrival',
           'is_delayed_120_departure', 'is_delayed_120_arrival',
           'delayed_group_d', 'delayed_group_a', 'is_delayed_arrival', 'is_delayed_departure']

    for col_name_dep in s_t:
        print(col_name_dep.upper())
        y = data[col_name_dep].copy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)
        form_df_benchmarks(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, name=col_name_dep)


if __name__ == "__main__":
    start_time = time.time()
    data = get_cleaned_data()
    iterate_over_dependent_to_csvs(data)
    print("--- %s seconds ---" % (time.time() - start_time))
    print('Successfully done!')
