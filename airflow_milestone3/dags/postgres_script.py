import pandas as pd
from sqlalchemy import create_engine


def task3():
    df = pd.read_csv('/opt/airflow/data/df_m2.csv')
    lookup = pd.read_csv('/opt/airflow/data/lookup_m2.csv')
    engine = create_engine('postgresql://root:root@pgdatabase:5432/accidents_etl')
    if engine.connect():
        print('connected succesfully')
    else:
        print('failed to connect')
    df.to_sql(name='UK_Accidents_2016', con=engine, if_exists='replace')
    lookup.to_sql(name='lookup_table', con=engine, if_exists='replace')
