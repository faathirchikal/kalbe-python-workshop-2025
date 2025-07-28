import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({'a':[1,2],
                   'b':[3,4]})
fig, ax = plt.subplots()
ax.plot(df['a'], df['b'])

st.title("My App Title")
st.header("Section Header")
st.subheader("Subsection")
st.text("This is plain text.")
st.markdown(
    "**Bold**, *italic*, and [links](https://streamlit.io)"
)
st.latex(r"\text{Latex time}: \int_{-\infty}^{\infty} f(x) dx")
st.dataframe(df)
st.pyplot(fig)
# st.plotly_chart(fig) # For Plotly
st.image('resource/langit_sore.jpg', caption='Senja', use_container_width=True)
st.video('https://www.youtube.com/watch?v=8aYdDLjTGeI')

if st.button('Click me'):
    st.write('Button clicked!')

agree = st.checkbox('I agree')
if agree:
    st.write('Great!')

choice = st.radio('Pick one', ['A', 'B', 'C'])

option = st.selectbox('Select', df['a'].unique())

options = st.multiselect('Choose', ['X', 'Y', 'Z'])

value = st.slider('Select a value', 0, 100, 50)

name = st.text_input('Your name')

num = st.number_input('Insert a number')

date = st.date_input('Pick a date')

t = st.time_input('Pick time')

uploaded = st.file_uploader('Upload a file')

sidebar_option = st.sidebar.selectbox(
    'Sidebar', ['Home', 'About']
)

with st.expander('More info'):
    st.write('Hidden content')

col1, col2 = st.columns(2)
col1.write('Column 1')
col2.write('Column 2')

st.metric(label="Temperature", value="70 °F", delta="+1.2 °F")

st.snow()