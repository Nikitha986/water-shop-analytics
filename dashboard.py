import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

st.title("💧 Water Shop Live Dashboard")

df = pd.read_csv("daily_report.csv")

st.metric("Customers Today", df.iloc[-1]["Customers"])
st.metric("Water Cans Sold", df.iloc[-1]["20L"])
st.metric("Unpaid", df.iloc[-1]["Unpaid"])

st.line_chart(df["Customers"])
st.bar_chart(df[["Cash","UPI","Unpaid"]])
