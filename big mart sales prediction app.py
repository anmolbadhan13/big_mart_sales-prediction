import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('Mart_model.sav')

# Load CSS
def load_css(css_file):
    with open(css_file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# App title

st.title("Big Mart Sales Prediction")
st.image("https://storage.googleapis.com/kaggle-datasets-images/2885406/4975802/63191e3407248852f08ccf8847b1f30d/dataset-card.png?t=2023-02-12-02-19-55", width=100)
# Load CSS
load_css("style.css")

# Sidebar for user input
st.sidebar.header("Input Features")

def user_input_features():
    item_weight = st.sidebar.number_input('Item Weight', min_value=0.0, max_value=100.0, value=10.0)
    item_visibility = st.sidebar.slider('Item Visibility', 0.0, 1.0, 0.05)
    item_mrp = st.sidebar.number_input('Item MRP', min_value=0.0, max_value=500.0, value=100.0)
    outlet_establishment_year = st.sidebar.slider('Outlet Establishment Year', 1985, 2024, 2000)
    outlet_size = st.sidebar.selectbox('Outlet Size', ('Small', 'Medium', 'High'))
    outlet_location_type = st.sidebar.selectbox('Outlet Location Type', ('Tier 1', 'Tier 2', 'Tier 3'))
    outlet_type = st.sidebar.selectbox('Outlet Type', ('Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store'))

    data = {'Item_Weight': item_weight,
            'Item_Visibility': item_visibility,
            'Item_MRP': item_mrp,
            'Outlet_Establishment_Year': outlet_establishment_year,
            'Outlet_Size': outlet_size,
            'Outlet_Location_Type': outlet_location_type,
            'Outlet_Type': outlet_type}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user input
st.subheader('User Input Features')
st.write(input_df)

# Encode categorical variables
def encode_features(df):
    df['Outlet_Size'] = df['Outlet_Size'].map({'Small': 0, 'Medium': 1, 'High': 2}).astype(int)
    df['Outlet_Location_Type'] = df['Outlet_Location_Type'].map({'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}).astype(int)
    df['Outlet_Type'] = df['Outlet_Type'].map({'Supermarket Type1': 0, 'Supermarket Type2': 1, 'Supermarket Type3': 2, 'Grocery Store': 3}).astype(int)
    return df

encoded_df = encode_features(input_df)

# Add missing features with default values
missing_features = {
    'Item_Identifier': 'FDX07',  # Example default value
    'Outlet_Identifier': 'OUT010',  # Example default value
    'Item_Fat_Content_Regular': 0,
    'Item_Type_Breads': 0,
    'Item_Type_Breakfast': 0,
    'Item_Type_Canned': 0,
    'Item_Type_Dairy': 0,
    'Item_Type_Frozen Foods': 0,
    'Item_Type_Fruits and Vegetables': 0,
    'Item_Type_Hard Drinks': 0,
    'Item_Type_Health and Hygiene': 0,
    'Item_Type_Household': 0,
    'Item_Type_Meat': 0,
    'Item_Type_Others': 0,
    'Item_Type_Seafood': 0,
    'Item_Type_Snack Foods': 0,
    'Item_Type_Soft Drinks': 0,
    'Item_Type_Starchy Foods': 0
}

for feature in missing_features:
    encoded_df[feature] = missing_features[feature]

# Convert categorical identifiers to numeric
unique_item_identifiers = {id: i for i, id in enumerate(encoded_df['Item_Identifier'].unique())}
unique_outlet_identifiers = {id: i for i, id in enumerate(encoded_df['Outlet_Identifier'].unique())}

encoded_df['Item_Identifier'] = encoded_df['Item_Identifier'].map(unique_item_identifiers).astype(int)
encoded_df['Outlet_Identifier'] = encoded_df['Outlet_Identifier'].map(unique_outlet_identifiers).astype(int)

# Ensure all data types are correct
encoded_df = encoded_df.astype({
    'Item_Weight': float,
    'Item_Visibility': float,
    'Item_MRP': float,
    'Outlet_Establishment_Year': int,
    'Item_Fat_Content_Regular': int,
    'Item_Type_Breads': int,
    'Item_Type_Breakfast': int,
    'Item_Type_Canned': int,
    'Item_Type_Dairy': int,
    'Item_Type_Frozen Foods': int,
    'Item_Type_Fruits and Vegetables': int,
    'Item_Type_Hard Drinks': int,
    'Item_Type_Health and Hygiene': int,
    'Item_Type_Household': int,
    'Item_Type_Meat': int,
    'Item_Type_Others': int,
    'Item_Type_Seafood': int,
    'Item_Type_Snack Foods': int,
    'Item_Type_Soft Drinks': int,
    'Item_Type_Starchy Foods': int
})

# Reorder columns to match model expectations
encoded_df = encoded_df[[
    'Item_Identifier', 'Item_Weight', 'Item_Visibility', 'Item_MRP',
    'Outlet_Identifier', 'Outlet_Establishment_Year', 'Outlet_Size',
    'Outlet_Location_Type', 'Outlet_Type', 'Item_Fat_Content_Regular',
    'Item_Type_Breads', 'Item_Type_Breakfast', 'Item_Type_Canned',
    'Item_Type_Dairy', 'Item_Type_Frozen Foods', 'Item_Type_Fruits and Vegetables',
    'Item_Type_Hard Drinks', 'Item_Type_Health and Hygiene', 'Item_Type_Household',
    'Item_Type_Meat', 'Item_Type_Others', 'Item_Type_Seafood',
    'Item_Type_Snack Foods', 'Item_Type_Soft Drinks', 'Item_Type_Starchy Foods'
]]

# Make prediction
try:
    prediction = model.predict(encoded_df)
    st.subheader('Predicted Sales')
    st.write(f"${prediction[0]:,.2f}")
except Exception as e:
    st.error(f"Error making prediction: {e}")