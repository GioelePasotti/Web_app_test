import streamlit as st
import pandas as pd
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split

col1, col2= st.beta_columns(2)
with col1:
    st.write("**Profile :** https://www.rstiwari.com")
with col2:
    st.write("**Blog :** https://tiwari11-rst.medium.com/") # Text/Title

st.title("Logistic Regression - Mnist Dataset")