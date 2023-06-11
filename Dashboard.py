import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler
from TrainData import preprocess_data
import numpy as np
app = dash.Dash()
server = app.server

BTC_train, BTC_valid, BTC_df_nse = preprocess_data("./BTC-USD.csv") #lay train data cua BTC
ETH_train, ETH_valid, ETH_df_nse = preprocess_data("./ETH-USD.csv") #lay train data cua ETH
ADA_train, ADA_valid, ADA_df_nse = preprocess_data("./ADA-USD.csv") #lay train data cua ADA


app.layout = html.Div([
   
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
   
    dcc.Tabs(id="tabs", children=[
       
        dcc.Tab(label='Bitcoin USD Stock Data',children=[
            html.Div([
                html.H2("Actual BTC closing price",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x=BTC_train.index,
                                y=BTC_train["Close"],
                                mode='markers'
                            )
                        ],
                        "layout":go.Layout(
                            title='Scatter Plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate (USD)'}
                        )
                    }
                ),
                html.H2("LSTM Predicted BTC closing price and compare",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x=BTC_valid.index,
                                y=BTC_valid["Close"],
                                mode='lines+markers',
                                line=dict(color="#0000ff"),
                                name="Close Price"
                            ),
                            go.Scatter(
                                x=BTC_valid.index,
                                y=BTC_valid["Predictions"],
                                mode='lines+markers',
                                line=dict(color="#ffe476"),
                                name="Predictions Price"
                            )
                        ],
                        "layout":go.Layout(
                            title='Scatter Plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate (USD)'}
                        )
                    }
                ),
                html.H2("High and Low BTC price",style={"textAlign": "center"}),
                dcc.Graph(
                    id="HighAndLow Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x=BTC_df_nse.index,
                                y=BTC_df_nse["High"],
                                mode='lines+markers',
                                line=dict(color="#0000ff"),
                                name="High Price"
                            ),
                            go.Scatter(
                                x=BTC_df_nse.index,
                                y=BTC_df_nse["Low"],
                                mode='lines+markers',
                                line=dict(color="#ffe476"),
                                name="Low Price"
                            )
                        ],
                        "layout":go.Layout(
                            title='Scatter Plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate (USD)'}
                        )
                    }
                ),          
            ])                
        ]),
        dcc.Tab(label='Ethereum USD Stock Data',children=[
            html.Div([
                html.H2("Actual ETH closing price",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x=ETH_train.index,
                                y=ETH_train["Close"],
                                mode='markers'
                            )
                        ],
                        "layout":go.Layout(
                            title='Scatter Plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate (USD)'}
                        )
                    }
                ),
                html.H2("LSTM Predicted ETH closing price and compare",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x=ETH_valid.index,
                                y=ETH_valid["Close"],
                                mode='lines+markers',
                                line=dict(color="#0000ff"),
                                name="Close Price"
                            ),
                            go.Scatter(
                                x=ETH_valid.index,
                                y=ETH_valid["Predictions"],
                                mode='lines+markers',
                                line=dict(color="#ffe476"),
                                name="Predictions Price"
                            )
                        ],
                        "layout":go.Layout(
                            title='Scatter Plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate (USD)'}
                        )
                    }
                ),
                html.H2("High and Low BTC price",style={"textAlign": "center"}),
                dcc.Graph(
                    id="HighAndLow Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x=ETH_df_nse.index,
                                y=ETH_df_nse["High"],
                                mode='lines+markers',
                                line=dict(color="#0000ff"),
                                name="High Price"
                            ),
                            go.Scatter(
                                x=ETH_df_nse.index,
                                y=ETH_df_nse["Low"],
                                mode='lines+markers',
                                line=dict(color="#ffe476"),
                                name="Low Price"
                            )
                        ],
                        "layout":go.Layout(
                            title='Scatter Plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate (USD)'}
                        )
                    }
                ),          
            ])                
        ]),
        dcc.Tab(label='Cardano USD Stock Data',children=[
            html.Div([
                html.H2("Actual ADA closing price",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Actual Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x=ADA_train.index,
                                y=ADA_train["Close"],
                                mode='markers'
                            )
                        ],
                        "layout":go.Layout(
                            title='Scatter Plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate (USD)'}
                        )
                    }
                ),
                html.H2("LSTM Predicted ADA closing price and compare",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x=ADA_valid.index,
                                y=ADA_valid["Close"],
                                mode='lines+markers',
                                line=dict(color="#0000ff"),
                                name="Close Price"
                            ),
                            go.Scatter(
                                x=ADA_valid.index,
                                y=ADA_valid["Predictions"],
                                mode='lines+markers',
                                line=dict(color="#ffe476"),
                                name="Predictions Price"
                            )
                        ],
                        "layout":go.Layout(
                            title='Scatter Plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate (USD)'}
                        )
                    }
                ),
                html.H2("High and Low BTC price",style={"textAlign": "center"}),
                dcc.Graph(
                    id="HighAndLow Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x=ADA_df_nse.index,
                                y=ADA_df_nse["High"],
                                mode='lines+markers',
                                line=dict(color="#0000ff"),
                                name="High Price"
                            ),
                            go.Scatter(
                                x=ADA_df_nse.index,
                                y=ADA_df_nse["Low"],
                                mode='lines+markers',
                                line=dict(color="#ffe476"),
                                name="Low Price"
                            )
                        ],
                        "layout":go.Layout(
                            title='Scatter Plot',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Closing Rate (USD)'}
                        )
                    }
                ),          
            ])                
        ]),

    ])
])
if __name__=='__main__':
    app.run_server(debug=True)