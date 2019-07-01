from datetime import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import warnings
from IPython.display import clear_output
from statsmodels.tsa.arima_model import ARIMA
from fbprophet import Prophet
import pandas as pd

def get_min(df, lower_bound):
    """
    Get minimum value of a dataframe, bigger than lower bound.

    :param df: Dataframe to get min from
    :type df: Pandas Dataframe
    :param lower_bound: Values smaller or equal are excluded from min
    :type lower_bound: int
    :return: Minimum
    """

    return df.loc[df > lower_bound].min()

def get_max(df, upper_bound):
    """
    Get maximum value of a dataframe, smaller than upper bound.

    :param df: Dataframe to get max from
    :type df: Pandas Dataframe
    :param upper_bound: Values bigger or equal are excluded from max
    :type upper_bound: int
    :return: Maximum
    """

    return df.loc[df < upper_bound].max()

def get_mean(df, lower_bound, upper_bound):
    """
    Get mean value of a dataframe, with respect to values between lower and upper bound.

    :param df: Dataframe to get mean from
    :type df: Pandas Dataframe
    :param lower_bound: Values smaller or equal are excluded from mean
    :type lower_bound: int
    :param upper_bound: Values bigger or equal are execluded from mean
    :type upper_bound: int
    :return: Mean
    """

    return df.loc[(df < upper_bound) & (df > lower_bound)].mean()

def generate_plt_ticks(nr_ticks, list):
    """
    Genereate list of indexes and list of labels to show on axis of matplotlib plot.

    :param nr_ticks: Amount of ticks to generate
    :type nr_ticks: int
    :param list: List of labels
    :type list: list
    :return: list of indexes and list of labels
    """

    all_indexes = len(list)-1
    dist = all_indexes/(nr_ticks-1)
    index_list = []
    for nr in range(nr_ticks):
        index_list.append(int(nr*dist))
    tick_list = []
    for index in index_list:
        tick_list.append(list[index])
    return index_list, tick_list


def format_text(text):
    """
    Transforms text to same format.
    Only numbers or digits allowed, space repalced with "-"
    Letters only upper case
    :param text: String to get formatted
    :type text: str
    :return: Formatted string
    """

    text = str(text)
    text = text.upper()
    text = text.replace(" ", "-")
    getVals = list([val for val in text if val.isalpha() or val.isnumeric() or (val == '-')])
    text = "".join(getVals)
    return text

def draw_city(name, lat, lng, map):
    """
    Draw marker for city on Basemap map.

    :param name: Name of the city
    :type name: str
    :param lat: Latitude location of the city
    :type lat: float
    :param lng: Longitude location of the city
    :type lng: float
    :param map: Map to draw cities on
    :type map: Basemap
    :return:
    """
    big_cities = {
        'Kaiserslautern': 'KL',
        'Mannheim': 'MA',
        'Karlsruhe': 'KA',
        'Ludwigshafen': 'LU',
    }
    x, y = map(lng, lat)
    if name in big_cities:
        plt.plot(x, y, 'ok', markersize=5)
        plt.text(x + 1000, y + 1000, big_cities[name], fontsize=10);
    else:
        plt.plot(x, y, 'or', markersize=2)


def time_stamp_to_date(time_stamp):
    """
    Converts time stamp string to datetime.

    :param time_stamp: Time stamp in format "%Y-%m-%d %H:%M:%S."
    :type time_stamp: str
    :return: datetime
    """

    time_stamp = time_stamp.split(".")[0]
    time = dt.strptime(time_stamp, "%Y-%m-%d %H:%M:%S")
    return time

def time_stamp_to_week(time_stamp):
    """
    Converts time stamp string to week in year.

    :param time_stamp: Time stamp in format "%Y-%m-%d %H:%M:%S."
    :type time_stamp: str
    :return: String of year and week in year
    """

    time_stamp = time_stamp.split(".")[0]
    time = dt.strptime(time_stamp, "%Y-%m-%d %H:%M:%S")
    week = time.isocalendar()[1]
    year = time.year
    return (str(year) + " Week: " + str(week))

def time_stamp_to_weekday(time_stamp):
    """
    Converts time stamp string to day in week

    :param time_stamp: Time stamp in format "%Y-%m-%d %H:%M:%S."
    :type time_stamp: str
    :return: number representing day in week
    """

    time_stamp = time_stamp.split(".")[0]
    time = dt.strptime(time_stamp, "%Y-%m-%d %H:%M:%S")
    weekday = time.weekday()
    return weekday

def time_stamp_to_hour(time_stamp):
    """
    Converts time stamp string to hour of day.

    :param time_stamp: Time stamp in format "%Y-%m-%d %H:%M:%S."
    :type time_stamp: str
    :return: number representing hour in day
    """

    time_stamp = time_stamp.split(".")[0]
    time = dt.strptime(time_stamp, "%Y-%m-%d %H:%M:%S")
    hour = time.hour
    return hour

def between_12_13(time_stamp):
    """
    Checks if time is between 12 and 13 o'clock.

    :param time_stamp: Time stamp in format "%Y-%m-%d %H:%M:%S."
    :type time_stamp: str
    :return: bool, true if time between 12 and 13 o'clock, else false
    """

    time_stamp = time_stamp.split(".")[0]
    time = dt.strptime(time_stamp, "%Y-%m-%d %H:%M:%S")
    return (time.hour == 12)

def between_7_20(time_stamp):
    """
    Checks if time is between 7 and 20 o'clock.

    :param time_stamp: Time stamp in format "%Y-%m-%d %H:%M:%S."
    :type time_stamp: str
    :return: bool, true if time between 7 and 20 o'clock, else false
    """

    time_stamp = time_stamp.split(".")[0]
    time = dt.strptime(time_stamp, "%Y-%m-%d %H:%M:%S")
    bet_7_20 = time.hour > 6 and time.hour < 21
    return bet_7_20

def time_stamp_to_month(time_stamp):
    """
    Converts time stamp to month in year.

    :param time_stamp: Time stamp in format "%Y-%m-%d %H:%M:%S."
    :type time_stamp: str
    :return: Number of month
    """

    time_stamp = time_stamp.split(".")[0]
    time = dt.strptime(time_stamp, "%Y-%m-%d %H:%M:%S")
    return time.month

def time_stamp_to_day(time_stamp):
    """
    Returns date in form "%Y-%m-%d"

    :param time_stamp: Time stamp in format "%Y-%m-%d %H:%M:%S."
    :type time_stamp: str
    :return: date
    """

    time = time_stamp.split(" ")[0]
    return time


def plot_time_series_pred(train_data, test_data, pred_data, title):
    """
    Plots a time series graph of training data followed by test and prediction data.
    """

    date_list = list(train_data.index) + list(test_data.index)
    train_range = list(range(len(train_data)))
    test_range = list(range(len(train_data), len(train_data) + len(test_data)))

    plt.plot(train_range, train_data.values, 'b', label="Training data")
    plt.plot(test_range, test_data.values, 'bo', label="Correct")
    plt.plot([train_range[-1], test_range[0]], [train_data.values[-1], test_data.values], 'b:')
    plt.plot(test_range, pred_data, 'yo', label="Prediction")
    plt.plot([train_range[-1], test_range[0]], [train_data.values[-1], pred_data], 'y:')
    x_index_list_ticks, x_label_list_ticks = generate_plt_ticks(6, date_list)
    plt.xticks(fontsize=8, rotation=30)
    plt.xticks(x_index_list_ticks, x_label_list_ticks)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.2f €'))
    plt.legend()
    plt.title(title)
    plt.show()


def plot_time_series_pred_prophet(train_data, test_data, pred_data, title):
    """
    Plots a time series graph of training data followed by test and prediction data.
    """

    date_list = list(train_data["ds"]) + list(test_data["ds"])
    train_range = list(range(len(train_data)))
    test_range = list(range(len(train_data), len(train_data) + len(test_data)))

    plt.plot(train_range, train_data["y"].values, 'b', label="Training data")
    plt.plot(test_range, test_data["y"].values, 'bo', label="Correct")
    plt.plot([train_range[-1], test_range[0]], [train_data["y"].values[-1], test_data["y"].values], 'b:')
    plt.plot(test_range, pred_data.tail(len(test_data))["yhat"], 'yo', label="Prediction")
    plt.plot([train_range[-1], test_range[0]], [train_data["y"].values[-1], pred_data.tail(len(test_data))["yhat"]], 'y:')
    x_index_list_ticks, x_label_list_ticks = generate_plt_ticks(6, date_list)
    plt.xticks(fontsize=8, rotation=30)
    plt.xticks(x_index_list_ticks, x_label_list_ticks)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.2f €'))
    plt.legend()
    plt.title(title)
    plt.show()

def suspress_ARIMA_warnigs():
    """
    Suspresses specific warnings thrown by the ARIMA module.
    """

    warnings.filterwarnings("ignore", message="Inverting hessian failed, no bse or cov_params available")
    warnings.filterwarnings("ignore", message="overflow encountered in exp")
    warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
    warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed to converge. Check mle_retvals")


def suspress_prophet_warnigs():
    """
    Suspresses specific warnings thrown by the ARIMA module.
    """

    warnings.filterwarnings("ignore", message="fbprophet:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.")
    warnings.filterwarnings("ignore", message="fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.")


def get_mad(gt, pred):
    """
    Calculates the Mean Average Deviation
    """

    gt_mean = gt.mean()
    mad = np.sum(np.abs((pred - gt_mean)))/pred.shape[0]
    return mad


def get_mape(gt, pred):
    """
    Calculates the Mean Absolute Percentage Error.
    """

    mape = 100*np.sum(np.abs((gt - pred)/gt))/pred.shape[0]
    return mape


def update_progress(progress):
    """
    Set progress bar to value of progress
    """

    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1
    block = int(round(bar_length * progress))

    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)

def apply_moving_average(train_data, test_data):
    """
    Apply simple moving average model to train_data.
    Make prediction of len of test_data
    """

    # Set parameters of ARIMA model:
    # 0: Order Autoregressive model
    # 0: degree of differencing
    # 4: order of moving average model
    # -> MA model only
    mw_model = ARIMA(train_data.values, order=(0, 0, 4))
    mw_model_fit = mw_model.fit(start_params=(0, 0, 0, 0, 0))
    pred = mw_model_fit.predict(start=len(train_data), end=(len(train_data) + len(test_data) - 1))
    return pred


def apply_fb_prophet(train_data, max_test_data_size):
    """
    Apply facebook prophet model to train_data.
    Make prediction of len of max_test_data_size
    """

    m = Prophet()
    m.fit(train_data)
    future_df = m.make_future_dataframe(periods=max_test_data_size)
    pred = m.predict(future_df)
    return pred