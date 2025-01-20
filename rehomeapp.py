import streamlit as st
import pandas as pd
import sklearn
from rehome_content import main # type: ignore

# Load the datasets
D = pd.read_csv("donor_dataset.csv")
C = pd.read_csv("charity_data.csv")

# Streamlit app title
st.title("Charity Recommendation System")

# Get user inputs
donor_id = st.number_input('Enter Donor ID:', min_value=0, max_value=len(D)-1, value=0)
scenario = st.selectbox('Select Scenario:', [1, 2, 3, 4])
radius = st.slider('Select Radius for Nearby Charity Search (miles):', min_value=1, max_value=100, value=5)

# Button to trigger the recommendation
if st.button('Get Recommendations'):
    st.write("Generating recommendations...")
    result = main(donor_id, scenario, radius)
    st.write(result)

