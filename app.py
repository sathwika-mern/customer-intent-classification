
import streamlit as st
import joblib
from pathlib import Path

# Configure the page
st.set_page_config(
    page_title="Customer Intent Classifier",
    page_icon="💬",
    layout="centered"
)

# Page heading
st.title("💬 Customer Intent Classifier")
st.write(
    "Enter a customer message to predict what the customer wants."
)

# Find and load the saved model
MODEL_PATH = Path(__file__).parent / "intent_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Place intent_model.pkl "
        "in the same folder as app.py."
    )
    st.stop()

model = load_model()

# Input field
st.subheader("Enter your customer message")

message = st.text_area(
    "Customer message",
    placeholder="Example: I want to reorder the item",
    height=120
)

# Example messages
st.write("**Try an example:**")

col1, col2 = st.columns(2)

with col1:
    if st.button("Cancel an order"):
        message = "I want to cancel my order"

with col2:
    if st.button("Reorder an item"):
        message = "I want to reorder the item"

# Predict button
if st.button("Predict Intent", type="primary"):
    if not message.strip():
        st.warning("Please enter a customer message.")
    else:
        prediction = model.predict([message.strip()])[0]

        st.success("Prediction completed!")
        st.subheader("Predicted Intent")
        st.markdown(f"### `{prediction}`")

st.divider()
st.caption("Customer Intent Classification | ML Project")
