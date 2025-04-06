import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import association_rules, apriori

# Set page configuration
st.set_page_config(page_title="Market Basket Analysis", page_icon="🍞", layout="wide")

# Title and description
st.title("Bakery Shop Item Recommender")
st.write("A Web App that helps recommend items to customers!")

# Load and process data
@st.cache_data
def load_data():
    data = pd.read_csv("bread basket.csv")
    data['date_time'] = pd.to_datetime(data['date_time'], format="%d-%m-%Y %H:%M")
    data["month"] = data['date_time'].dt.month_name()
    data["day"] = data['date_time'].dt.day_name()
    return data

data_depl = load_data()

# Sidebar for user input
st.sidebar.header('User Input')
st.sidebar.write("Use these widgets to input values")

item = st.sidebar.selectbox("Item", data_depl["Item"].unique())
period_day = st.sidebar.selectbox('Period Day', ['Morning', 'Afternoon', 'Evening', 'Night'])
weekday_weekend = st.sidebar.selectbox('Weekday / Weekend', ['Weekend', 'Weekday'])
month = st.sidebar.select_slider("Month", data_depl["month"].unique())
day = st.sidebar.select_slider('Day', data_depl["day"].unique())

# Filter data based on user input
def get_data(item, period_day, weekday_weekend, month, day):
    data = data_depl.copy()
    filtered = data.loc[
        (data["Item"] == item) & 
        (data["period_day"] == period_day.lower()) &
        (data["weekday_weekend"] == weekday_weekend.lower()) &
        (data["month"] == month) &
        (data["day"] == day)
    ]
    return filtered

# Display filtered data based on user choice
filtered_data = get_data(item, period_day, weekday_weekend, month, day)

st.write("### Filtered Transactions:")
st.dataframe(filtered_data)

# Encoding the data for Market Basket Analysis
def encode_units(x):
    return 0 if x <= 0 else 1

# Apriori and Association Rules Calculation
if not filtered_data.empty:
    st.write("### Running Apriori Algorithm for Recommendations...")

    # Get the original data without filtering down too much for Apriori
    # We will use only item filtering for Apriori, so we don't exclude too many transactions
    apriori_data = data_depl.copy()

    item_count = apriori_data.groupby(["Transaction", "Item"])["Item"].count().reset_index(name="Count")
    item_count_pivot = item_count.pivot_table(index='Transaction', columns='Item', values='Count', aggfunc='sum').fillna(0)
    item_count_pivot = item_count_pivot.applymap(encode_units)

    # Perform Apriori analysis
    support = 0.01  # Minimum support level (adjustable)
    frequent_items = apriori(item_count_pivot, min_support=support, use_colnames=True)

    # Generate association rules
    if not frequent_items.empty:
        metric = "lift"
        min_threshold = 1
        rules = association_rules(frequent_items, metric=metric, min_threshold=min_threshold)
        rules.sort_values('confidence', ascending=False, inplace=True)

        # Display recommendations
        st.write("### Recommendation:")

        # Ensure we handle the case where no rules are generated
        if not rules.empty:
            # Check if any rules match the selected item in the antecedents
            recommendations = rules[rules["antecedents"].apply(lambda x: item in list(x))]

            if not recommendations.empty:
                recommended_item = list(recommendations.iloc[0]["consequents"])[0]
                st.write(f"Customer who buys **{item}**, also buys **{recommended_item}**!")
            else:
                st.write("No recommendations available for this item.")
        else:
            st.write("No association rules generated from the dataset.")
    else:
        st.write("No frequent itemsets found.")
else:
    st.write("No transactions found with the selected filters.")
