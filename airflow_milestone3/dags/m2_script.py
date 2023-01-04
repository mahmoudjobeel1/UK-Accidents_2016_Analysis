import pandas as pd
import reverse_geocoder as rg
from sklearn import preprocessing


def task2():
    df = pd.read_csv('/opt/airflow/data/df_m1.csv')
    lookup = pd.read_csv('/opt/airflow/data/lookup_m1.csv')
    df = add_feature(df)
    df, lookup = label_encode_feature(df, lookup, 'city')
    df.to_csv('/opt/airflow/data/df_m2.csv', index=False)
    lookup.to_csv('/opt/airflow/data/lookup_m2.csv', index=False)


def add_feature(df):
    coordinates = list(zip(df.latitude, df.longitude))
    results = rg.search(coordinates, mode=1)  # default mode = 2
    cities = list(map(lambda res: res['admin2'], results))
    df['city'] = cities
    return df


# encode column 'city' and add it to lookup table you created in task1
def label_encode_feature(df, lookup, feature_name):
    feature_labels = preprocessing.LabelEncoder().fit_transform(df[feature_name])
    feature_values = df[feature_name].to_numpy()

    feature_name_set = set()

    for i in range(len(df[feature_name])):
        if feature_values[i] in feature_name_set:
            continue
        lookup.loc[len(lookup)] = {"feature_value": feature_name + "::" + feature_values[i], "code": feature_labels[i]}
        feature_name_set.add(feature_values[i])

    df[feature_name] = feature_labels
    return df, lookup
