###############################################################
# Business Problem
###############################################################

# FLO wants to establish a roadmap for its sales and marketing activities.
# To be able to plan for the medium to long-term future of the company, it is necessary to estimate the potential value that existing customers will bring to the company in the future.


###############################################################
# Dataset Information
###############################################################

# master_id: Unique customer ids
# order_channel : Which channel of the shopping platform was used (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : Last shopping channel
# first_order_date : First shopping date of customer
# last_order_date : Last shopping date of customer
# last_order_date_online : Last online shopping date of customer
# last_order_date_offline : Last offline shopping date of customer
# order_num_total_ever_online : Total number of orders on online by customer
# order_num_total_ever_offline : Total number of orders on offline by customer
# customer_value_total_ever_offline : Total spending of customer on offline shopping
# customer_value_total_ever_online : Total spending of customer on online shopping
# interested_in_categories_12 : Categories that customer bought product in last 12 months


###############################################################
# Data Preperation
###############################################################
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler

df_ = pd.read_csv("flo_data_20k.csv")
df  = df_.copy()


def outlier_thresholds(dataframe, variable):
    """
    To suppress outliers, threshold values are determined.
    :param dataframe: pandas.core.frame.DataFrame
    :param variable: pandas.core.series.Series
    :return:
    """
    quartile_1 = dataframe[variable].quantile(0.01)
    quartile_3 = dataframe[variable].quantile(0.99)

    interquantile_range = quartile_3 - quartile_1

    up_limit = round(interquantile_range * 1.5 + quartile_3)
    low_limit = round(interquantile_range * 1.5 + quartile_1)

    return up_limit, low_limit

def replace_with_thresholds(dataframe, variable):
    """
    Replace outlier values with thresholds.
    :param dataframe: pandas.core.frame.DataFrame
    :param variable: pandas.core.series.Series
    :return:
    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable)

    # low_limit replace
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit

    # up_limit replace
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Adjust thresholds for outlier values.

outlier_variables = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online"]

for cols in outlier_variables:
    replace_with_thresholds(df, cols)

# Function for data preprocessing.

def preprocess(dataframe):
    """
    Preprocesses  data before RFM segmentation.

    :param dataframe: pandas.core.frame.DataFrame
    :return: pandas.core.frame.DataFrame
    """

    # Omnichannel refers to customers who make purchases both online and offline.
    df["Omnichannel_order"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]
    df["Omnichannel_spend"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

    # Change dataype of date variables into datetime
    date_columns = df.loc[:, df.columns.str.contains("date")]
    df[date_columns.columns] = date_columns.apply(pd.to_datetime)

    # The distribution of the number of customers in shopping channels, the average number of products purchased, and the average spending.
    df.groupby(["order_channel"]).agg({"master_id": "count",
                                       "Omnichannel_order": "mean",
                                       "Omnichannel_spend": "mean"})
    return dataframe

###############################################################
# Create CLTV Datatype
###############################################################

today_date = dt.datetime(2021, 6, 2)

cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"] - df["first_order_date"]).astype("timedelta64[D]"))/7
cltv_df["T_weekly"] = ((today_date - df["first_order_date"]).astype("timedelta64[D]")) / 7
cltv_df["frequency"] = df["omnichannel_order"]
cltv_df["monetary_cltv_avg"] = df["omnichannel_value"] / df["omnichannel_order"]

###############################################################
# Build BG/NBD, Gamma-Gamma Models and calculations for 6 month
###############################################################

# Build BG/NBD Model
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"],
        cltv_df["recency_cltv_weekly"],
        cltv_df["T_weekly"])

# Excpected sales from customers in 3 months

cltv_df["exp_sales_3_month"] = bgf.predict(12,
                                           cltv_df["frequency"],
                                           cltv_df["recency_cltv_weekly"],
                                           cltv_df["T_weekly"])
# Expected sales from customers in 6 months

cltv_df["exp_sales_6_month"] = bgf.predict(24,
                                           cltv_df["frequency"],
                                           cltv_df["recency_cltv_weekly"],
                                           cltv_df["T_weekly"])

# Build Gamma Gamma Submodel

ggf = GammaGammaFitter(penalizer_coef= 0.001)

ggf.fit(cltv_df["frequency"],
        cltv_df["monetary_cltv_avg"])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                       cltv_df["monetary_cltv_avg"])

# CLTV calculation for 6 months of period

cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,
                                              cltv_df["frequency"],
                                              cltv_df["recency_cltv_weekly"],
                                              cltv_df["T_weekly"],
                                              cltv_df["monetary_cltv_avg"],
                                              time = 6,
                                              freq = "W")


###############################################################
# Customer segmentation based on CLTV
###############################################################

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels = ["D", "C", "B", "A"])

# Mean values of recency, frequency and monetary values of customers based on their segments

cltv_df.groupby("cltv_segment")["recency_cltv_weekly", "frequency", "monetary_cltv_avg"].agg("mean")


###############################################################
# # Function for whole CLTV prediction process to improve functionality
###############################################################

def create_cltv_df(dataframe):
    """
    Create 3 months and 6 months of CLTV predictions with Gamma Gamma submodel and BG/NBD Model.
    :param dataframe: pandas.core.series.Series
    :return: pandas.core.series.Series
    """
    columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline","customer_value_total_ever_online"]
    for col in columns:
        replace_with_thresholds(dataframe, col)

    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    dataframe = dataframe[~(dataframe["customer_value_total"] == 0) | (dataframe["order_num_total"] == 0)]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)

    # CLTV datatype
    dataframe["last_order_date"].max()  # 2021-05-30
    analysis_date = dt.datetime(2021, 6, 1)
    cltv_df = pd.DataFrame()
    cltv_df["customer_id"] = dataframe["master_id"]
    cltv_df["recency_cltv_weekly"] = ((dataframe["last_order_date"] - dataframe["first_order_date"]).astype('timedelta64[D]')) / 7
    cltv_df["T_weekly"] = ((analysis_date - dataframe["first_order_date"]).astype('timedelta64[D]')) / 7
    cltv_df["frequency"] = dataframe["order_num_total"]
    cltv_df["monetary_cltv_avg"] = dataframe["customer_value_total"] / dataframe["order_num_total"]
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

    # BG-NBD Model
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly'])
    cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])
    cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])

    # # Gamma-Gamma Submodel
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
    cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                           cltv_df['monetary_cltv_avg'])

    # CLTV prediction
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'],
                                       cltv_df['monetary_cltv_avg'],
                                       time=6,
                                       freq="W",
                                       discount_rate=0.01)
    cltv_df["cltv"] = cltv

    # CLTV segmentation
    cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_df



