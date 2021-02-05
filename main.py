import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


def main():

    company = 'FB'
    data = load_data(company)

    prediction_days = 60
    x_train, y_train, scaler = prepare_data(data, prediction_days)

    model = build_model(x_train, y_train, units=50, epochs=25, batch_size=32)

    actual_prices, predicted_prices, model_inputs = test_model(company, data, model, prediction_days, scaler)

    # plot the Test Predictions
    plot_test_prediction(actual_prices, company, predicted_prices)
    
    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs+1), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(f'Prediction: {prediction}')


##############################################

def plot_test_prediction(actual_prices, company, predicted_prices):
    plt.plot(actual_prices, color='black', label=f'Actual {company} Price')
    plt.plot(predicted_prices, color='green', label=f'Predicted {company} Price')
    plt.title(f'{company} Share Price')
    plt.xlabel('Time')
    plt.ylabel(f'{company} Share Price')
    plt.legend()
    plt.show()


def test_model(company, data, model, prediction_days, scaler):
    # test the model accuracy on existing data

    # Load Test Data
    model_inputs, actual_prices = load_test_data(company, data, prediction_days, scaler)

    # make predictions on Test Data
    predicted_prices = predict_data(model, model_inputs, prediction_days, scaler)

    return actual_prices, predicted_prices, model_inputs


def predict_data(model, model_inputs, prediction_days, scaler):
    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    return predicted_prices


def load_test_data(company, data, prediction_days, scaler):
    test_start = dt.datetime(2020, 1, 1)
    test_end = dt.datetime.now()
    test_data = web.DataReader(company, 'yahoo', test_start, test_end)
    actual_prices = test_data['Close'].values
    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)
    return model_inputs, actual_prices


def build_model(x_train, y_train, units, epochs, batch_size):
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Prediction of the next closing value
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs, batch_size)
    return model


def prepare_data(data, prediction_days):
    # prepare Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    x_train = []
    y_train = []
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train, scaler


def load_data(company):
    start = dt.datetime(2012, 1, 1)
    end = dt.datetime(2020, 1, 1)
    data = web.DataReader(company, 'yahoo', start, end)
    return data

##############################################


if __name__ == "__main__":
    main()
