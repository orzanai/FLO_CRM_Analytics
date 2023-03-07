###############################################################
# Business Problem
###############################################################

# FLO wants to segment its customers and develop marketing strategies based on these segments.
# For this purpose, customer behaviors will be defined and groups will be created based on these behavior patterns.

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

# Data Understanding

import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()

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
    #The distribution of the number of customers in shopping channels, the average number of products purchased, and the average spending.
    df.groupby(["order_channel"]).agg({"master_id": "count",
                                       "Omnichannel_order": "mean",
                                       "Omnichannel_spend": "mean"})
    return dataframe

# Calculate RFM Metrics

df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 2)

rfm = df.groupby("master_id").agg({"last_order_date": lambda x: (today_date - x.max()).days,
                             "Omnichannel_order": lambda x: x,
                             "Omnichannel_spend": lambda x: x})

rfm.columns = ["recency", "frequency", "monetary"]

# Create RF and RFM Scores

rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels = [1, 2, 3, 4, 5])

# Finalize RF Score value with Recency and Frequency
rfm["rfm_score"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

# RF Define Segments from RF Scores

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['rfm_score'].replace(seg_map, regex=True)


# Check mean of recency, frequency and monetary values of segments

rfm.groupby("segment").agg({"recency": "mean",
                            "frequency":"mean",
                            "monetary": "mean"})

rfm.groupby("segment")["recency", "frequency", "monetary"].agg("mean")


# FLO is introducing a new women's shoe brand. The prices of the brand's products are above the general customer preferences.
# Therefore, it is desired to communicate with customers who are interested in the brand's promotion and product sales through targeted communication.
# Communication will be established specifically with customers who shop in the women's category and are in the Champions and Loyal_Customers classes.
# The ID numbers of these customers were exported to a CSV file named "new_brand_target_cust_id"."

rfm_updated = rfm.merge(df[['interested_in_categories_12', 'master_id']], on='master_id')

export_df = rfm_updated[(rfm_updated["segment"].isin(["champions", "loyal_customers"])) & (rfm_updated["interested_in_categories_12"].str.contains("KADIN"))]

export_df["master_id"].to_csv("new_brand_target_cust_id.cvs")

# A discount of nearly 40% is planned for Men's and Children's products.
# Those interested in the categories related to this discount, past good customers who haven't shopped for a long time, customers who should not be lost, dormant customers, and new customers are specially targeted.
# The selection of suitable profiled customers and their IDs were exported as output to the "discount_target_cust_ids" CSV file.

# Only category of "[ERKEK, COCUK]"
discount_df = rfm_updated[(rfm_updated["segment"].isin(["hibernating", "new_customers", "cant_loose"])) &
                          ((rfm_updated["interested_in_categories_12"] == "[ERKEK]") |
                          (rfm_updated["interested_in_categories_12"] == "[COCUK]"))]

# The categories that includes "[ERKEK]" ve "[COCUK]"
discount_df = rfm_updated[(rfm_updated["segment"].isin(["hibernating", "new_customers", "cant_loose"])) &
                          ((rfm_updated["interested_in_categories_12"].str.contains("ERKEK")) |
                          (rfm_updated["interested_in_categories_12"].str.contains("COCUK")))]

discount_df.to_csv("discount_target_cust_ids.csv")


# Function for whole RFM Segmentation process to improve functionality

def get_rfm(path, csv=False):
    """
    Creates RFM dataframe with segments from given dataset path.
    Exports results as cvs file if csv parameter's value True.
    :param path: str
    :param csv: boolean
    :return: pandas.core.frame.DataFrame
    """
    df_ = pd.read_csv(path)


    df["Omnichannel_order"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]
    df["Omnichannel_spend"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]


    date_columns = df.loc[:, df.columns.str.contains("date")]
    df[date_columns.columns] = date_columns.apply(pd.to_datetime)

    # RFM Metriklerinin Hesaplanması

    # recency, frequency, monetary

    df["last_order_date"].max()
    today_date = dt.datetime(2021, 6, 2)

    rfm = df.groupby("master_id").agg({"last_order_date": lambda x: (today_date - x.max()).days,
                                       "Omnichannel_order": lambda x: x,
                                      "Omnichannel_spend": lambda x: x})

    rfm.columns = ["recency", "frequency", "monetary"]
    rfm.reset_index(inplace=True)
    # RF ve RFM Skorlarının Hesaplanması

    rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

    rfm["rfm_score"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_Risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm['segment'] = rfm['rfm_score'].replace(seg_map, regex=True)

    if csv:
        rfm.to_csv("rfm.csv")
    return rfm
