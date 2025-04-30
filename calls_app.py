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

# Define pages
def intro():
    st.title("Analysis of Baltimore 911 Calls")
    st.markdown("by Cassie Chou and Julia Martin | Data Science for Public Health 4th Term Capstone")
    st.write("This app provides a descriptive analysis of 911 calls in Baltimore, Maryland, and predicts the priority of calls based on various features inlcuding date, time, and neighborhood.")
    st.write("The data used in this app is sourced from Open Baltimore API 2024 911 Calls for Service, which contains 1.6 million call records and information about each call including date, time, neighborhood, priority, and description of emergency.")
    st.write("Use the sidebar to navigate between the Descriptive Analysis and Prediction pages.")


def page_1():
    st.title("Descriptive Analysis")

    # Read in data from Open Baltimore API 2024 911 Calls for Service
    calls = pd.read_csv("911_Calls_for_Service_2024.csv")

    # Read in geo data for community statistical areas
    csa_data = gpd.read_file("csa_shapes.shp")

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

    st.write("Data is organized into 55 **Community Statistical Areas (CSA)**, which are clusters of neighborhoods in Baltimore organized around Census tract boundaries.")
    st.write("Calls are coded as **Non-Emergency (1)**, **Low (2)**, **Medium (3)**, **High (4)**, or **Emergency (5)**. The average priority score is calculated by dividing the sum of call priorities by the total number of calls in each CSA.")

    #fig1.update_geos(fitbounds="locations", visible=False)
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

def page_2():
    st.title("Prediction")
    st.write("This is the Prediction page.")

# Sidebar navigation
page = st.sidebar.radio("Select a Page", ["Welcome", "Descriptive Analysis", "Prediction"])

# Render the selected page
if page == "Welcome":
    intro()
if page == "Descriptive Analysis":
    page_1()
elif page == "Prediction":
    page_2()