import streamlit as st
st.title('Big mart sales analysis app')
import pandas as pd
df=pd.read_csv('Data.csv') #for displaying df
st.dataframe(df)

#a line chart
st.line_chart(df[['Age','Salary']])
st.line_chart(df[['Age']])

# display completed on right side
col1, col2=st.columns(2)
with col2:
	st.write('completed') 
with col1:
	st.write('left side') 

# code using python
# ui
# documentation
