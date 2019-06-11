from datetime import datetime as dt
import matplotlib.pyplot as plt

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

