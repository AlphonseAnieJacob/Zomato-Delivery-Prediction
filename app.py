import streamlit as st
import pandas as pd
import joblib

# Load models
linear_model = joblib.load("linear_model.pkl")
logistic_model = joblib.load("logistic_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Page title
st.set_page_config(
    page_title="ABC Ltd Delivery Predictor",
    page_icon="📦"
)

st.title("📦 ABC Ltd Delivery Prediction Tool")
st.write("Predict delivery time and identify deliveries that may be late.")

st.divider()

# User inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Delivery person's age",
        min_value=18,
        max_value=60,
        value=30
    )

    rating = st.number_input(
        "Delivery person's rating",
        min_value=1.0,
        max_value=5.0,
        value=4.5,
        step=0.1
    )

    weather = st.selectbox(
        "Weather",
        ["Sunny", "Cloudy", "Fog", "Stormy", "Sandstorms", "Windy"]
    )

    traffic = st.selectbox(
        "Road traffic",
        ["Low", "Medium", "High", "Jam"]
    )

    vehicle_condition = st.selectbox(
        "Vehicle condition",
        [0, 1, 2, 3]
    )

    vehicle_type = st.selectbox(
        "Vehicle type",
        ["motorcycle", "scooter", "electric_scooter"]
    )

with col2:
    order_type = st.selectbox(
        "Order type",
        ["Snack", "Meal", "Drinks", "Buffet"]
    )

    multiple_deliveries = st.selectbox(
        "Multiple deliveries",
        [0, 1, 2, 3]
    )

    festival = st.selectbox(
        "Festival",
        ["No", "Yes"]
    )

    city = st.selectbox(
        "City",
        ["Metropolitan", "Urban", "Semi-Urban"]
    )

    order_date = st.date_input("Order date")

    order_time = st.time_input("Order time")

    picked_time = st.time_input("Order picked-up time")


st.divider()

st.subheader("📍 Location")

col3, col4 = st.columns(2)

with col3:
    restaurant_latitude = st.number_input(
        "Restaurant latitude",
        value=20.0,
        format="%.6f"
    )

    restaurant_longitude = st.number_input(
        "Restaurant longitude",
        value=77.0,
        format="%.6f"
    )

with col4:
    delivery_latitude = st.number_input(
        "Delivery latitude",
        value=20.0,
        format="%.6f"
    )

    delivery_longitude = st.number_input(
        "Delivery longitude",
        value=77.0,
        format="%.6f"
    )


# Prediction button
if st.button("🔮 Predict Delivery", use_container_width=True):

    # Create input dataframe
    input_data = pd.DataFrame({
        "Delivery_person_Age": [age],
        "Delivery_person_Ratings": [rating],
        "Restaurant_latitude": [restaurant_latitude],
        "Restaurant_longitude": [restaurant_longitude],
        "Delivery_location_latitude": [delivery_latitude],
        "Delivery_location_longitude": [delivery_longitude],
        "Order_Date": [pd.Timestamp(order_date)],
        "Time_Orderd": [str(order_time)],
        "Time_Order_picked": [str(picked_time)],
        "Weather_conditions": [weather],
        "Road_traffic_density": [traffic],
        "Vehicle_condition": [vehicle_condition],
        "Type_of_order": [order_type],
        "Type_of_vehicle": [vehicle_type],
        "multiple_deliveries": [multiple_deliveries],
        "Festival": [festival],
        "City": [city]
    })

    # Create date features
    input_data["Order_Day"] = input_data["Order_Date"].dt.day
    input_data["Order_Month"] = input_data["Order_Date"].dt.month
    input_data["Order_Weekday"] = input_data["Order_Date"].dt.weekday

    # Create time features
    input_data["Order_Hour"] = pd.to_datetime(
        input_data["Time_Orderd"],
        errors="coerce"
    ).dt.hour

    input_data["Order_Picked_Hour"] = pd.to_datetime(
        input_data["Time_Order_picked"],
        errors="coerce"
    ).dt.hour

    # Remove original date/time columns
    input_data = input_data.drop(
        columns=[
            "Order_Date",
            "Time_Orderd",
            "Time_Order_picked"
        ]
    )

    # Convert categories to numbers
    input_data = pd.get_dummies(
        input_data,
        drop_first=True
    )

    # Make input match the model's 30 columns
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Make predictions
    predicted_time = linear_model.predict(input_data)[0]

    late_probability = logistic_model.predict_proba(input_data)[0][1]

    late_prediction = logistic_model.predict(input_data)[0]

    # Display results
    st.divider()
    st.subheader("📊 Prediction")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Predicted delivery time",
            f"{predicted_time:.0f} minutes"
        )

    with result_col2:
        st.metric(
            "Late probability",
            f"{late_probability * 100:.1f}%"
        )

    if late_prediction == 1:
        st.error("⚠️ This delivery is predicted to be LATE.")
    else:
        st.success("✅ This delivery is predicted to be ON TIME.")
