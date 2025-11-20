import streamlit as st
from .util import * 


def preview_image():

    pass

def preview_video():

    pass

def preview_log():

    st.markdown(f"<h3 style='text-align: left;'>Data Type: {st.session_state.log_type}</h3>", unsafe_allow_html=True)
    
    text = st.session_state.text_input
    if text:
        st.markdown(f"<h3 style='text-align: left;'>Text</h3>", unsafe_allow_html=True)
        # text_area_style()
        st.text_area("Input Text", text, disabled=False, label_visibility="collapsed")
    else:
        st.markdown(f"<h3 style='text-align: left;'>No Input Text</h3>", unsafe_allow_html=True)
    
    uploaded_files = st.session_state.uploaded_file
    
    if uploaded_files:
        st.markdown("### Files")
        for file in uploaded_files:
            st.markdown("---")
            col1, col2 = st.columns([3, 1])  # col1: 파일 이름, col2: 버튼
            with col1:
                st.write(f"📄 {file.name}")
            with col2:
                st.download_button(
                    label="Download",
                    data=file.getvalue(),
                    file_name=file.name,
                    mime=file.type
                )
        
            file.seek(0)
            content = file.read().decode("utf-8", errors="ignore")
            # text_area_style()
            st.text_area(f"{file.name}", content, height=400, disabled=False, label_visibility="collapsed", key=f"File_{i}")


def preview_time():
    
    st.markdown(f"<h3 style='text-align: left;'>Time Method: {st.session_state.time_method_labels[st.session_state.time_type]}</h3>", unsafe_allow_html=True)

    text = st.session_state.text_input
    if text:
        st.markdown(f"<h3 style='text-align: left;'>Text</h3>", unsafe_allow_html=True)
        # text_area_style()
        st.text_area("Input Text", text, disabled=False, label_visibility="collapsed")
    else:
        st.markdown(f"<h3 style='text-align: left;'>No Input Text</h3>", unsafe_allow_html=True)
    
    uploaded_files = st.session_state.uploaded_file

    if uploaded_files:
        st.markdown("### Files & Visualize")
        for file in uploaded_files:
            st.markdown("---")
            col1, col2 = st.columns([3, 1])  # col1: 파일 이름, col2: 버튼
            with col1:
                st.write(f"📄 {file.name}")
            with col2:
                st.download_button(
                    label="Download",
                    data=file.getvalue(),
                    file_name=file.name,
                    mime=file.type
                )

            try:
                values = make_list(file)
                image = make_ts_image(len(values), values)
                st.image(image, caption=f"{file.name} Visualization")
            except Exception as e:
                st.error(f"Error while processing input file name:  {file.name}\n{e}")
                st.session_state.input_error = True