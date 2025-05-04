import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import folium
import contextily as cx
import geopandas as gpd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import pickle
from model_def import EnhancedPredictor

# Define pages
def intro():
    st.title("Analysis of Baltimore 911 Calls")
    st.markdown("by Cassie Chou and Julia Martin | Data Science for Public Health 4th Term Capstone")
    url = "https://bmore-open-data-baltimore.hub.arcgis.com/datasets/911-calls-for-service-2024-1/explore"
    st.write("This app provides a descriptive analysis of 911 calls in Baltimore, Maryland, and predicts the priority of calls based on various features inlcuding date, time, and neighborhood.")
    st.write("The data used in this app is sourced from [Open Baltimore API 2024 911 Calls for Service](%s)" % url)
    st.write("The call records contain 1.6 million call records and information about each call including date, time, neighborhood, priority, and description of emergency. The goal of this project is understand where and when 911 calls are being made and where and when high priority calls are being made so that that the appropriate emergency response teams can be prepared to respond to calls.")
    st.write("Use the sidebar to navigate between the Descriptive Analysis and Prediction pages.")
    st.write("**Contents**")
    st.write("1. Descriptive Analysis")
    st.write("  - Spatial and temporal trends in 911 calls and priority level")
    st.write("2. Prediction Page 1")


def page_1():
    st.title("Descriptive Analysis")

    @st.cache_data
    def cache_load(url):
        data = pd.read_csv(url)
        return data

    @st.cache_data
    def cache_load_geo(url):
        data = gpd.read_file(url)
        return data

    calls = cache_load("911_Calls_for_Service_2024.csv")

    csa_data = cache_load_geo("csa_shapes.shp")

    # subset calls data where priority is out of service
    calls_out_of_service = calls[calls['priority'] == 'Out of Service']
    calls_out_of_service.head()

    # create new column for call priority code
    calls['priority_code'] = np.select(
    [calls['priority'] == 'Non-Emergency', calls['priority'] == 'Low', calls['priority'] == 'Medium', calls['priority'] == 'High', calls['priority'] == 'Emergency', calls['priority'] == 'Out of Service'],
    [1, 2, 3, 4, 5, 0]
    )

    # aggregate calls by CSA and sum of priority code and total number of calls
    calls_agg = calls.groupby('Community_Statistical_Areas').agg(
    total_calls=('priority_code', 'count'),
    total_priority=('priority_code', 'sum')
    ).reset_index()

    calls_agg['priority_score'] = calls_agg['total_priority'] / calls_agg['total_calls']

    # merge calls_agg with csa_data
    csa_data = csa_data.merge(calls_agg, left_on='CSA2010', right_on='Community_Statistical_Areas', how='left')

    @st.cache_data
    def create_plot_1():
        fig1 = px.choropleth_map(
            csa_data,
            geojson=csa_data.geometry,
            locations=csa_data.index,
            color='priority_score',
            color_continuous_scale='OrRd',
            map_style="carto-positron",
            zoom=9.8,
            center={"lat": 39.2905, "lon": -76.6104},
            opacity=0.5,
            labels={"CSA2010": 'CSA', 'priority_score': 'Average Priority'},
            hover_name="CSA2010",  # Main label for hover
            hover_data={
                "priority_score": True
            },
            ).update_traces(hovertemplate=None)  # Disable default hover template

        fig1.update_layout(
            title = " Priority of 911 Calls by Community Statistical Area",
            title_x = 0.15,
            annotations=[
                dict(
                text="1 = Non-Emergency, 2 = Low, 3 = Medium, 4 = High, 5 = Emergency",  # Add your subtitle here
                x=0.15,  # Center the subtitle
                y=-.1,  # Position below the title
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14, color="black") # Customize font size and color
            )]
        )
        return fig1

    st.write("Data is organized into 55 **Community Statistical Areas (CSA)**, which are clusters of neighborhoods in Baltimore organized around Census tract boundaries.")
    st.write("Calls are coded as **Non-Emergency (1)**, **Low (2)**, **Medium (3)**, **High (4)**, or **Emergency (5)**. The average priority score is calculated by dividing the sum of call priorities by the total number of calls in each CSA.")

    #fig1.update_geos(fitbounds="locations", visible=False)
    fig1 = create_plot_1()
    st.plotly_chart(fig1)


    # CSA vs. Calls per 1000 people (Heat Map) 


    # read in data with population
    population_data = pd.read_csv("Total_Population.csv")

    # merge with csa_data
    csa_data = csa_data.merge(population_data, left_on='CSA2010', right_on='CSA2010', how='left')

    # create new column for total calls per 1000 people
    csa_data['calls_per_1000'] = (csa_data['total_calls'] / csa_data['tpop20']) * 1000

    fig2 = px.choropleth_map(
        csa_data,
        geojson=csa_data.geometry,
        locations=csa_data.index,
        color='calls_per_1000',
        color_continuous_scale='OrRd',
        map_style="carto-positron",
        zoom=9.8,
        center={"lat": 39.2905, "lon": -76.6104},
        opacity=0.5,
        labels={"CSA2010": 'CSA', 'calls_per_1000': 'Annual Calls per 1000 People'},
        hover_name="CSA2010",  # Main label for hover
        hover_data={
            "calls_per_1000": True
        },
    ).update_traces(hovertemplate=None)  # Disable default hover template

    fig2.update_layout(
    title = "Call Density by Community Statistical Area",
    title_x = 0.2,
    )

    st.plotly_chart(fig2)
    # group data by priority and count the number of calls
    calls_grouped = calls.groupby('priority').size().reset_index(name='frequency')

    # create horizontal stacked bar chart in plotly of priority frequency
    fig3 = px.bar(
        calls_grouped,
        x="frequency",  # Replace with the column representing the frequency of calls
        y="priority",   # Replace with the column representing priority levels
        color="priority",  # Color by priority levels
        orientation="h",  # Horizontal bar chart
        title="Priority Frequency",
        labels={"frequency": "Frequency of Calls", "priority": "Priority Level"}
    )

    fig3.update_traces(
        textposition="outside"  # Position text labels outside the bars
    )
    # Show the plot
    st.plotly_chart(fig3)

    st.write(
        "Over 2/3 of calls are classified as Non-Emergency calls, examples of which include noise complaints or business checks (when businesses call to request that police come to patrol the area due to security concerns or loitering). In fact, 529,826 (almost 1/3) of all calls in 2024 were business checks.")

    st.write("Only 412 calls were classified as Emergency calls, locations of which are shown below, however there were several emergency calls with missing location data.") 

    calls_emergency = calls[calls['priority'] == 'Emergency']

    # group by community statistical area and count the number of calls
    calls_emergency_grouped = calls_emergency.groupby('Community_Statistical_Areas').size().reset_index(name='frequency')

    # merge calls_agg with csa_data
    csa_data_em = csa_data.merge(calls_emergency_grouped, left_on='CSA2010', right_on='Community_Statistical_Areas', how='left')

    # plot frequency of emergency calls by community statistical area
    fig4 = px.choropleth_map(
        csa_data_em,
        geojson=csa_data_em.geometry,
        locations=csa_data_em.index,
        color='frequency',
        color_continuous_scale='OrRd',
        map_style="carto-positron",
        zoom=9.8,
        center={"lat": 39.2905, "lon": -76.6104},
        opacity=0.5,
        labels={"CSA2010": 'CSA', 'frequency': '# Annual Emergency Calls'},
        hover_name="CSA2010",  # Main label for hover
        hover_data={
            "frequency": True
        },
    ).update_traces(hovertemplate=None)  # Disable default hover template
    fig4.update_layout(
        title = "Emergency Calls by Community Statistical Area",
        title_x = 0.2,
    )

    st.plotly_chart(fig4)

    # extract month from date
    calls['month'] = pd.to_datetime(calls['callDateTime']).dt.month
    # group by month and count the number of calls
    calls_monthly = calls.groupby('month').size().reset_index(name='frequency')
    # create line chart of calls by month
    fig5 = px.line(
        calls_monthly,
        x="month",  # Replace with the column representing the month
        y="frequency",  # Replace with the column representing the frequency of calls
        title="911 Calls by Month",
        labels={"month": "Month", "frequency": "Frequency of Calls"}
    )

    # Show the plot
    st.plotly_chart(fig5)

    # extract time from date
    calls['time'] = pd.to_datetime(calls['callDateTime']).dt.hour
    # group by time and count the number of calls
    calls_hourly = calls.groupby('time').size().reset_index(name='frequency')
    # create line chart of calls by hour
    fig6 = px.line(
        calls_hourly,
        x="time",  # Replace with the column representing the hour
        y="frequency",  # Replace with the column representing the frequency of calls
        title="911 Calls by Hour",
        labels={"time": "Time (24 hour clock)", "frequency": "Frequency of Calls"}
    )
    # Show the plot
    st.plotly_chart(fig6)

    st.write("The graphs displaying temporal trends in 911 calls show that October had by far the most calls of any month in 2024, and there is an influx of calls in the late morning and evening hours.")

def page_2():
    st.title("Prediction")
    st.write("Please input Community Area, Date, and Time to get a predicted priority call level.")

    # Read in Model and Preprocessors
    with open('priority_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('preprocessors.pkl', 'rb') as f:
        preprocessors = pickle.load(f)
    scaler = preprocessors['scaler']           # StandardScaler/MinMaxScaler
    le_priority = preprocessors['le_priority'] # LabelEncoder for priority
    le_csa = preprocessors['le_csa']
    le_neighborhood = preprocessors['le_neighborhood']
    le_description = preprocessors['le_description']

    # Read in Data
    csa = pd.read_csv("csa.csv")
    neighborhood = pd.read_csv("neighborhood.csv")
    description = pd.read_csv("description.csv")

    # Get Inputs
    community_area = st.selectbox("Select Community Statistical Area", options=csa, index = 0)
    neighborhood_val = st.selectbox("Select Neighborhood", options=neighborhood, index = 0)
    descrip_val = st.selectbox("Select Description", options=description, index = 0)
    date = st.date_input("Select Date", value=pd.to_datetime("2024-01-01"))
    time = st.time_input("Select Time", value=pd.to_datetime("12:00").time())
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_of_week = st.selectbox("Select Day of Week", options=days_of_week, index = 0)

    le_dayOTW = LabelEncoder()
    le_dayOTW.classes_ = np.array(days_of_week)
    day_of_week = le_dayOTW.transform([day_of_week])
    community_area = le_csa.transform([[community_area]])
    neighborhood_val = le_neighborhood.transform([[neighborhood_val]])
    descrip_val = le_description.transform([[descrip_val]])

    x_input = pd.DataFrame({
        'month': pd.to_datetime(date).month,
        'day': pd.to_datetime(date).day,
        'time': int(time.strftime("%H%M%S")),
        'day_of_week': day_of_week
    })

    features = ['month', 'day', 'time']
    x_input[features] = scaler.transform(x_input[features])
    x_input = x_input.join(pd.DataFrame(neighborhood_val, columns = le_neighborhood.categories_))
    x_input = x_input.join(pd.DataFrame(descrip_val, columns=le_description.categories_))
    x_input_torch = torch.from_numpy(x_input.values.astype(np.float32))

    model.eval()
    with torch.no_grad():
        logits = model(x_input_torch)
        probs = torch.softmax(logits, dim = 1)
        preds = torch.argmax(probs, dim = 1)

    preds = le_priority.inverse_transform(preds.numpy())
    st.write("Predicted Priority Level: ", preds.item())

# Sidebar navigation
page = st.sidebar.radio("Select a Page", ["Welcome", "Descriptive Analysis", "Prediction"])

# Render the selected page
if page == "Welcome":
    intro()
if page == "Descriptive Analysis":
    page_1()
elif page == "Prediction":
    page_2()