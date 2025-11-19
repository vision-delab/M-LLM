import streamlit as st
from pathlib import Path
from PIL import Image
from .time_result_content import time_method1, time_method2, time_method3
from .log_result_content import log_pre, log_unpre

def image_result(result: dict):
    st.write("Prediction: ", result.get("prediction"))
    st.write("Score: ", result.get("score"))
    st.write("Text received: ", result.get("text_received"))
    st.write("File name: ", result.get("file_name"))
    # st.write("Prediction:", result.get("prediction"))
    # st.write("Score:", result.get("score"))
    # st.write("Text received:", result.get("text_received"))
    # st.write("File name:", result.get("file_name"))

    # result_image_path = result.get("result_image_path")
    # if result_image_path:
    #     img = Image.open(result_image_path)
    #     st.image(img, caption="Result Image", use_column_width=True)
    
    
def log_result(result: dict):

    if result is None:
        st.write("Some error occurred. No result to display.")
        return
    
    if 'error' in result.keys():
        st.write(f"Error: {result['error']}")
        return
    
    log_type = result.get("log_type", [])
    
    if st.session_state.text_input or log_type[0] == 'others':
        log_unpre(result)
    else:
        log_pre(result)
    
def time_result(result: dict):
    
    if result is None:
        st.write("Some error occurred. No result to display.")
        return
    
    if 'error' in result.keys():
        st.write(f"Error: {result['error']}")
        return
    
    method = result.get("method", [])
    
    if method[0] == 'Method1':
        time_method1(result)
    elif method[0] == 'Method2':
        time_method2(result)
    elif method[0] == 'Method3':
        time_method3(result)
    
    
def video_result(result: dict):
    st.write("Prediction: video")
    st.write("Score: video")
    st.write("Text received: video")
    st.write("File name: video")
    # st.write("Prediction:", result.get("prediction"))
    # st.write("Score:", result.get("score"))
    # st.write("Text received:", result.get("text_received"))
    # st.write("File name:", result.get("file_name"))

    # result_image_path = result.get("result_image_path")
    # if result_image_path:
    #     img = Image.open(result_image_path)
    #     st.image(img, caption="Result Image", use_column_width=True)