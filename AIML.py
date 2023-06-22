import json
import numpy as np
# Data to be included in the JSON file
data = {
    "intents": [
        {"tag": "greetings",
         "patterns": ["Hey", "Hello", "Whats up?", "Hi"],
         "responses": ["Hello sir!", "Hey Aryan!"]
         },
        {"tag": "plot_chart",
         "patterns": ["I want you to plot a stock price!", "Please plot a chart"],
         "responses": ["Plotting chart"]
         },
        {"tag": "add_portfolio",
         "patterns": ["Add a stock to my portfolio", "Add a stock"],
         "responses": ["Adding stock"]
         },
        {"tag": "remove_portfolio",
         "patterns": ["Remove a stock from my portfolio", "Remove a stock"],
         "responses": ["Removing stock"]
         },
        {"tag": "show_portfolio",
         "patterns": ["Show me my investments", "Show me my portfolio"],
         "responses": ["Showing portfolio"]
         },
        {"tag": "stock_price",
         "patterns": ["I want to know the price of a stock!", "Show me the price of a stock"],
         "responses": ["Showing the price of the stock"]
         },
        {"tag": "portfolio_worth",
         "patterns": ["How much is my portfolio worth?", "What is the value of my portfolio"],
         "responses": ["Portfolio Worth"]
         },
        {"tag": "portfolio_gains",
         "patterns": ["How is my portfolio performing?", "How are my gains"],
         "responses": ["Portfolio Gains"]
         },
        {"tag": "prediction",
         "patterns": ["predict stock prices", "predict stocks"],
         "responses": ["Predicting Future Prices"]
         },
        {"tag": "Bye",
         "patterns": ["Bye", "goodbye", "stop", "I have to go", "end"],
         "responses": ["Goodbye"]
         }

    ]
}

# Convert data to JSON string
json_data = json.dumps(data, indent=4)

# Write JSON string to a file
with open("data.json", "w") as json_file:
    json_file.write(json_data)

print("JSON file created successfully.")

from neuralintents import GenericAssistant

import matplotlib.pyplot as plt
import pandas_datareader as web
import mplfinance as mpf
import pickle
import sys
import datetime as dt
import pandas as pd
import yfinance as yf


# portfolio = { 'AAPL' : 20, 'TSLA': 5, 'GS': 10}
#
# with open('portfolio.pkl','wb') as f:
#   pickle.dump(portfolio,f)
def save_portfolio(portfolio):
  with open('portfolio.pkl','wb') as f:
    pickle.dump(portfolio,f)

def add_portfolio():
  ticker=input("Which stock do you want to add:")
  amount=input("How many shares do you want to add:")
  with open('portfolio.pkl','rb') as f:
      portfolio=pickle.load(f)
      if ticker in portfolio.keys():
        portfolio[ticker]+=int(amount)
      else:
        portfolio[ticker]=int(amount)
  save_portfolio(portfolio)

def remove_portfolio():
  ticker=input("Which stock do you want to sell:")
  amount=input("How many shares do you want to sell")
  with open('portfolio.pkl','rb') as f:
      portfolio = pickle.load(f)
      if ticker in portfolio.keys():
        if amount<=portfolio[ticker]:
          portfolio[ticker]-=int(amount)
        else:
          print("You don't have enough shares")
      else:
        print(f"You don't own any shares of {ticker}")
  save_portfolio(portfolio)

def show_portfolio():
  print("Your portfolio:")
  with open('portfolio.pkl', 'rb') as f:
      portfolio = pickle.load(f)
      for ticker in portfolio.keys():
        print(f"You own {portfolio[ticker]} shares of {ticker}")


def portfolio_worth():
  sum=0
  with open('portfolio.pkl', 'rb') as f:
      portfolio = pickle.load(f)
      for ticker in portfolio.keys():
        data=web.DataReader(ticker,'yahoo')
        price=data['Close'].iloc[-1]
        sum+=price
      print(f"Your portfolio is worth {sum} USD")


def portfolio_gains():
  starting_date=input("Enter a date for comparison (YYYY-MM-DD):")
  sum_now=0
  sum_then=0
  with open('portfolio.pkl', 'rb') as f:
      portfolio = pickle.load(f)
      try:
        for ticker in portfolio.keys():
          data=web.DataReader(ticker,'yahoo')
          price_now=data['Close'].iloc[-1]
          price_then=data.loc[data.index==starting_date]['Close'].values[0]
          sum_now+=price_now
          sum_then+=price_then
        print(f"Relative Gains : {((sum_now-sum_then)/sum_then)*100}%")
        print(f"Absolute Gains : {sum_now-sum_then} USD")
      except IndexError:
        print("There was no trading on this day!")



def plot_chart():
  ticker = input("Choose a ticker symbol:")
  starting_string = input("Choose a starting date (DD/MM/YYYY):")
  plt.style.use('dark_background')
  start = dt.datetime.strptime(starting_string, "%d/%m/%Y")
  end = dt.datetime.now()
  data = yf.download(ticker, start=start, end=end)
  colors=mpf.make_marketcolors(up='#00ff00',down='#ff0000',wick='inherit',edge='inherit',volume='in')
  mpf_style=mpf.make_mpf_style(base_mpf_style='nightclouds',marketcolors=colors)
  mpf.plot(data,type='candle',style=mpf_style,volume=True)

def stock_price():
    ticker = input("Choose a ticker symbol:")
    data = web.DataReader(ticker, 'yahoo')
    price = data['Close'].iloc[-1]
    print(f"Present price of the stock : {price}")

def predict():
    ticker = input("Choose a ticker symbol:")
    data = yf.download(ticker)

    # Dividing the dataset columns into X and y

    X = data.iloc[:, 1:4].values
    y = data.iloc[:, 4].values

    y = y.reshape(-1, 1)

    # Using Standard Scaler Library for Standardisation

    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)
    y = sc_X.fit_transform(y)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

    from sklearn.linear_model import LinearRegression
    y_test1 = sc_X.inverse_transform(y_test)



    from sklearn.tree import DecisionTreeRegressor

    regressor3 = DecisionTreeRegressor(random_state=0)
    regressor3.fit(X_train, y_train)
    y_pred3 = regressor3.predict(X_test)
    y_pred3 = sc_X.inverse_transform(y_pred3.reshape(-1, 1))
    a = np.arange(0, min(len(y_pred3), len(y_test1)))

    print(y_pred3[1])
    print(y_test1[1])

    plt.figure(figsize=(16, 9))
    plt.scatter(a, y_pred3, color='red')
    plt.scatter(a, y_test1, color='purple')
    plt.legend(labels=['y_pred', 'y_test'])
    plt.ylabel('Price USD')
    plt.xlabel('Trading days')
    plt.title('Test Values VS Predicted Values-- Decision Tree Regression')
    plt.savefig('DTR.png')
    plt.show()

    from sklearn.metrics import r2_score, mean_squared_error
    print("Decision Tree Regression :  \t{}".format(r2_score(y_test1, y_pred3)))


def bye():
  print("Goodbye")
  sys.exit(0)

mappings={
    "plot_chart":plot_chart,
    "add_portfolio":add_portfolio,
    "remove_portfolio":remove_portfolio,
    "portfolio_worth":portfolio_worth,
    "portfolio_gains":portfolio_gains,
    "bye":bye,
    "show_portfolio":show_portfolio,
    "stock_price":stock_price,
    "prediction":predict
}
assistant=GenericAssistant("data.json",mappings,"financial_assistant_model")
assistant.train_model()
assistant.save_model()


while True:
  message=input("enter:")
  assistant.request(message)
