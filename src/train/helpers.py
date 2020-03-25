import pandas as pd
from dateutil.relativedelta import relativedelta
import holidays
from math import sqrt


RU_HOLIDAYS = holidays.Russia()


def get_holidays_count(month_start):
    month_start = pd.to_datetime(month_start)
    month_end = month_start + relativedelta(months=1)
    return len(RU_HOLIDAYS[month_start:month_end])


def get_rmse(y_actual, y_predicted):
    return sqrt(mean_squared_error(y_actual, y_predicted))
