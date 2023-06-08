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
import numpy as np
app = dash.Dash()
server = app.server
scaler=MinMaxScaler(feature_range=(0,1))
df_nse = pd.read_csv("./BTC-USD.csv")
df_nse["Date"]=pd.to_datetime(df_nse.Date,format="%Y-%m-%d")
df_nse.index=df_nse['Date']
data=df_nse.sort_index(ascending=True,axis=0)
new_data=pd.DataFrame(index=range(0,len(df_nse)),columns=['Date','Close'])
for i in range(0,len(data)):
    new_data["Date"][i]=data['Date'][i]
    new_data["Close"][i]=data["Close"][i]
print(new_data.head())
new_data.index=new_data.Date
new_data.drop("Date",axis=1,inplace=True)
dataset=new_data.values
train=dataset[0:1100,:]
valid=dataset[1100:,:]
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)
model=load_model("saved_model.h5")
inputs=new_data[len(new_data)-len(valid)-60:].values
inputs=inputs.reshape(-1,1)
inputs=scaler.transform(inputs)
X_test=[]
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
closing_price=model.predict(X_test)
closing_price=scaler.inverse_transform(closing_price)
train=new_data[:1100]
valid=new_data[1100:]
valid['Predictions']=closing_price


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
                                x=train.index,
                                y=train["Close"],
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
                html.H2("LSTM Predicted BTC closing price and Compare",style={"textAlign": "center"}),
                dcc.Graph(
                    id="Predicted Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x=valid.index,
                                y=valid["Close"],
                                mode='lines+markers',
                                line=dict(color="#0000ff"),
                                name="Close Price"
                            ),
                            go.Scatter(
                                x=valid.index,
                                y=valid["Predictions"],
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
                                x=df_nse.index,
                                y=df_nse["High"],
                                mode='lines+markers',
                                line=dict(color="#0000ff"),
                                name="High Price"
                            ),
                            go.Scatter(
                                x=df_nse.index,
                                y=df_nse["Low"],
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