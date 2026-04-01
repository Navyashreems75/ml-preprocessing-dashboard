# Button
import streamlit as st
import pandas as pd

df=pd.read_csv('Data.csv')

# create the session for line chart and one more for bar chart
# one more for bar chart
if 'click_line' not in st.session_state:
	st.session_state.click_line=False
if 'click_bar' not in st.session_state:
	st.session_state.click_bar=False

#KPI=Key Performance Index or metrics

cl1,cl2,cl3=st.columns(3)
cl1.metric('Median Salary',df.Salary.median())
cl2.metric('Mean Age',df.Age.mean())
cl3.metric('Country count',df.Country.count())
cl1.metric('unique Country',df.Country.nunique())

col1, col2=st.columns(2)
with col1:
	if st.button('click_line'):
		st.session_state.click_line=True
with col2:
	if st.button('click_bar'):
		st.session_state.click_bar=True
with col1:
	if st.session_state.click_line:
		st.line_chart(df['Age'])
with col2:
	if st.session_state.click_bar:
		st.bar_chart(df['Country'].value_counts())
		