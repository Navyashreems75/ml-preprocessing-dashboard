import streamlit as st
import pandas as pd
st.title('Text elements')
st.header('Main heading')
st.subheader('Sub Heading')
st.write('Normal Fonts')
st.markdown('**BOLD**')
st.markdown("*Italic*")
st.markdown('**bold fonts**|*Italix*|code')

# use button to display df
# after clicking button it should display df

df=pd.read_csv('Data.csv')

if st.button("click for df"):
	st.success("button clicked")
	st.dataframe(df)
	
#line chart on left side and bar on right
#add one more button for line chart 
col1, col2=st.columns(2)
with col1:
	if st.button('click for line chart'):
		st.success("Line Chart")
		st.line_chart(df[['Age','Salary']])

#add one more button for bar chart 
with col2:

	if st.button('click for Bar chart'):
		st.success("Bar Chart")
		st.bar_chart(df['Salary'])
