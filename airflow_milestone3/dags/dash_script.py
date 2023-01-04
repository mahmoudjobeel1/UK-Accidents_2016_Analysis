import plotly.express as px
# from dash import Dash, dcc, html, Input, Output

import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import seaborn as sns
import matplotlib.pyplot as plt


def task4():
    df = pd.read_csv('/opt/airflow/data/df_m2.csv')
    lookup = pd.read_csv('/opt/airflow/data/lookup_m2.csv')
    create_dash(df,lookup)


def create_dash(df,lookup):
    app = dash.Dash()
    app.layout = html.Div([
        html.H1("Web Application Dashboards with Dash", style={'text-align': 'center'}),
        html.Br(),
        html.H1("UK Accidents 2016 dataset", style={'text-align': 'center'}),

        html.Br(),
        html.Div(),
        dcc.Graph(figure=graph1(df,lookup)),

        html.Br(),
        html.Div(),
        dcc.Graph(figure=graph2(df,lookup)),

        html.Br(),
        html.Div(),
        dcc.Graph(figure=graph3(df,lookup)),

        html.Br(),
        html.Div(),
        dcc.Graph(figure=graph4(df,lookup)),

        html.Br(),
        html.Div(),
        dcc.Graph(figure=graph5(df,lookup)),


    ])
    app.run_server(host="0.0.0.0")

def ScatterAverage(df, columnName1, columnName2):
    average = df.groupby([columnName1])[columnName2].sum()/ df.groupby([columnName1])[columnName2].count()
    fig_title="Relationship between" + columnName1+" and " + columnName2
    fig = px.scatter(x=average.index, y=average)
    fig.update_layout(
    title=fig_title,
    xaxis_title=columnName1,
    yaxis_title=columnName2)
    return fig

def ScatterSum(df, columnName1, columnName2):
    count = df.groupby([columnName1])[columnName2].sum()
    fig_title="Relationship between" + columnName1+" and " + columnName2
    fig = px.scatter(x=count.index, y=count)
    fig.update_layout(
    title=fig_title,
    xaxis_title=columnName1,
    yaxis_title=columnName2)
    return fig

def graph1(df,lookup):
    col1_name="junction_detail"
    col2_name="number_of_vehicles"
    replace_dataframe_from_lookup(df,lookup,col1_name)
    replace_dataframe_from_lookup(df,lookup,col2_name)
    return ScatterAverage(df,col1_name, col2_name)


def graph2(df,lookup):
    col1_name="road_type"
    col2_name="number_of_casualties"
    replace_dataframe_from_lookup(df,lookup,col1_name)
    replace_dataframe_from_lookup(df,lookup,col2_name)
    return ScatterAverage(df,col1_name, col2_name)
   



def graph3(df,lookup):
    col1_name="junction_control"
    col2_name="number_of_casualties"
    replace_dataframe_from_lookup(df,lookup,col1_name)
    replace_dataframe_from_lookup(df,lookup,col2_name)
    return ScatterAverage(df,col1_name, col2_name)
   



def graph4(df,lookup):
    col1_name="speed_limit"
    col2_name="number_of_casualties"
    replace_dataframe_from_lookup(df,lookup,col1_name)
    replace_dataframe_from_lookup(df,lookup,col2_name)
    return ScatterAverage(df,col1_name, col2_name)
    

def graph5(df,lookup):
    col1_name="number_of_vehicles"
    col2_name="number_of_casualties"
    replace_dataframe_from_lookup(df,lookup,col1_name)
    replace_dataframe_from_lookup(df,lookup,col2_name)
    return ScatterAverage(df,col1_name, col2_name)
   

def restore_values_lookup(lookup,col_name):
    lookup2=lookup[lookup['feature_value'].str.match(col_name)]
    if len(lookup2)==0:
        return "null"
    lookup2['feature_value']=lookup2['feature_value'].str.split('::',1).str[1]
    lookup2.to_csv('/opt/airflow/data/test.csv', index=False)
    values_dict={}
    for index, row in lookup2.iterrows():
        values_dict[row['code']]=row['feature_value']
    return {col_name:values_dict}

def replace_dataframe_from_lookup(df,lookup,col_name):
    values_dict=restore_values_lookup(lookup,col_name)
    if values_dict=='null':
        pass
    df.replace(values_dict)
    

