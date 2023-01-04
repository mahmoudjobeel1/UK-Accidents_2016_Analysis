import pandas as pd
import numpy as np
from m2_script import label_encode_feature


def task1():
    filename = '/opt/airflow/data/UK_Accidents_2016.csv'
    df = pd.read_csv(filename)
    df_clean = clean(df)
    df_trans, lookup = transform(df_clean)
    df_trans.to_csv('/opt/airflow/data/df_m1.csv', index=False)
    lookup.to_csv('/opt/airflow/data/lookup_m1.csv', index=False)


# don't transform longitude and latitude as i will be using them in task 2
# lookup table columns are ['feature_value', 'code']
def transform(df):
    df_trans = df.copy()
    lookup = pd.DataFrame(columns=['feature_value', 'code'])

    discretize_date(df_trans)
    feature_extraction(df_trans)

    df_trans,lookup = apply_label_encoded(df_trans, lookup)

    return df_trans, lookup


def discretize_date(df):
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df['time']= pd.to_datetime(df['time'], format='%H:%M')
    df['Week number'] = df['date'].dt.isocalendar().week


def apply_label_encoded(df, lookup):
    labels= df.select_dtypes(include = "object").columns.tolist()
    labels_to_delete=['accident_reference', 'lsoa_of_accident_location','date','time','first_road_number','second_road_number']

    for label in labels_to_delete:
        if label in labels:
            labels.remove(label)

    for label in labels:
        df,lookup=label_encode_feature(df,lookup,label)
    
    return df,lookup
    # label_encoded_values = {'accident_severity': {'Slight': 1, 'Serious': 2, 'Fatal': 3},
    #                         'light_conditions': {'Daylight': 1, 'Darkness - lights lit': 2,
    #                                              'Darkness - lights unlit': 3, 'Darkness - no lighting': 4,
    #                                              'Darkness - lighting unknown': 0},
    #                         'weather_conditions': {'Fine no high winds': 1, 'Fine + high winds': 2,
    #                                                'Raining no high winds': 3, 'Raining + high winds': 4,
    #                                                'Snowing no high winds': 5, 'Snowing + high winds': 6,
    #                                                'Fog or mist': 7, 'Other': 0},
    #                         'road_surface_conditions': {'Dry': 1, 'Wet or damp': 2, 'Flood over 3cm. deep': 3,
    #                                                     'Frost or ice': 4, 'Snow': 5},
    #                         'did_police_officer_attend_scene_of_accident': {'Yes': 1, "No": 0},
    #                         'trunk_road_flag': {'Trunk (Roads managed by Highways England)': 1, 'Non-trunk': 0}}

    # for primary_key in label_encoded_values.keys():
    #     for key, value in label_encoded_values[primary_key].items():
    #         lookup.loc[len(lookup)] = {"feature_value": primary_key + "::" + key, "code": value}


    # return df.replace(label_encoded_values)


def feature_extraction(df):
    add_weather_acc_happened_weekend(df)
    add_acc_day_time(df)


def add_weather_acc_happened_weekend(df):
    df["accident_on_weekend"] = np.where((df["day_of_week"] == "Saturday") | (df["day_of_week"] == "Sunday"), 1, 0)


def add_acc_day_time(df):
    df.loc[df['time'].dt.hour < 12, ['day_time']] = 'Morning'
    df.loc[(df['time'].dt.hour >= 12) & (df['time'].dt.hour < 19), ['day_time']] = 'Afternoon'
    df.loc[(df['time'].dt.hour >= 19) & (df['time'].dt.hour < 24), ['day_time']] = 'Night'


def clean(df):
    df_clean = df.replace(['Data missing or out of range', 'unknown (self reported)'], np.nan)
    df_clean = remove_rows_with_nan_less_than(df_clean, threshold=1)
    for col_name in ['road_type', 'weather_conditions', 'trunk_road_flag']:
        df_clean[col_name] = replace_nan_with_mode(df_clean, col_name)
    for col_name in ['junction_control', 'second_road_number']:
        df_clean[col_name] = df_clean[col_name].fillna('None')
    df_clean.second_road_class = df_clean.second_road_class.replace('-1', 'None')
    col_name = 'did_police_officer_attend_scene_of_accident'
    df_clean[col_name] = df_clean[col_name].replace(
        'No - accident was reported using a self completion  form (self rep only)', 'No')
    df_clean = df_clean.drop(['accident_index', 'accident_year', 'location_easting_osgr', 'location_northing_osgr'],
                             axis=1)
    all_cols_except_ref = df_clean.columns.to_list().remove('accident_reference')
    df_clean = df_clean.drop_duplicates(all_cols_except_ref)
    return df_clean


def remove_rows_with_nan_less_than(dataset: pd.DataFrame, threshold: float = 0):
    cnt = dataset.isna().sum()
    cnt = cnt[cnt > 0]
    cnt = cnt * 100 / len(dataset)
    cnt = cnt[cnt < threshold]
    cols = cnt.index.to_list()
    return dataset.dropna(axis='index', subset=cols)


def replace_nan_with_mode(dataset: pd.DataFrame, col_name):
    return dataset[col_name].fillna(dataset[col_name].mode()[0])
