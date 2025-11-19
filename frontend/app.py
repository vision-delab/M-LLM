import streamlit as st
import requests
from pathlib import Path
from utils import image_result, video_result, log_result, time_result, preview_image, preview_video, preview_log, preview_time
import json
from datetime import datetime

st.set_page_config(page_title="M-LLM Anomaly Detection", page_icon="🔍")

HISTORY_DIR = Path(__file__).parent.parent / "history"

# 기본 설정값 초기화
if "page" not in st.session_state:
    st.session_state.page = "input"
if "title" not in st.session_state:
    st.session_state.title = None    
if "model_type" not in st.session_state:
    st.session_state.model_type = "Image"
if "text_input" not in st.session_state:
    st.session_state.text_input = ""
st.session_state.input_error = False
     
# 결과 저장용
if "result" not in st.session_state:
    st.session_state.result = {}

# History 선택용
if "title_selected" not in st.session_state:
    st.session_state.title_selected = None
    
# Log를 위한 설정값
if "log_type" not in st.session_state:
    st.session_state.log_type = 'others'

# Time을 위한 설정값
if "time_type" not in st.session_state:
    st.session_state.time_type = 'Method1'
st.session_state.time_method_values = ["Method1", "Method2"]
st.session_state.time_method_labels = {
    "Method1": "Qwen",
    "Method2": "Qwen with Frequency Segment Modeling",
}

###############################################
# INPUT PAGE
###############################################
def render_input_page():
    st.markdown("<h1 style='text-align: center;'>Find Anomalys with M-LLM</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Upload your data</h2>", unsafe_allow_html=True)

    st.session_state.title = st.text_input("Enter title for this run")
    
    st.session_state.model_type = st.selectbox(
        "Select Anomaly Detection Type",
        ["Image", "Video", "Log", "Time"],
        index=["Image", "Video", "Log", "Time"].index(st.session_state.model_type)
    )
    
    if st.session_state.model_type == 'Log':
        st.session_state.log_type = st.selectbox(
            "Select Log Data Type",
            ["HDFS", "BGL", "Thunderbird", "others"],
            index=["HDFS", "BGL", "Thunderbird", "others"].index(st.session_state.log_type)
        )
    elif st.session_state.model_type == 'Time':
        st.session_state.time_type = st.selectbox(
            "Select Time-series detection Method",
            st.session_state.time_method_values,
            index=st.session_state.time_method_values.index(st.session_state.time_type),
            format_func=lambda x: st.session_state.time_method_labels[x]
        )

    st.session_state.text_input = st.text_area(
        "Enter keyword or prompt for anomaly detection",
        value=st.session_state.text_input,
        height=150,
        placeholder="Type here..."
    )

    if st.session_state.model_type == "Time":
        st.session_state.uploaded_file = st.file_uploader(
            "Upload file",
            type=["txt", "csv", "pkl"],
            accept_multiple_files=True
        )
    elif st.session_state.model_type == "Log":
        st.session_state.uploaded_file = st.file_uploader(
            "Upload file",
            type=["txt", "log"],
            accept_multiple_files=True
        )
    else:
        st.session_state.uploaded_file = st.file_uploader(
            "Upload file",
            type=["png", "jpg", "mp4", "avi", "txt", "csv", "log"],
            accept_multiple_files=True
        )

    cols = st.columns([1]*5)
    with cols[2]:
        if st.button("Continue"):
            errors = []
            if not st.session_state.title:
                errors.append("Please enter a title for this run")
            if not st.session_state.uploaded_file:
                errors.append("Please upload at least one file")

            if errors:
                st.session_state.errors = errors
            else:
                st.session_state.page = "preview"
                st.rerun()

    # 에러 메시지 전체 폭 + 가운데 정렬
    if st.session_state.get("errors"):
        for e in st.session_state.errors:
            st.markdown(
                f'''
                <div style="
                    color: #b71c1c; 
                    background-color: #ffcdd2; 
                    border: 1px solid #f44336; 
                    border-radius: 8px;
                    padding: 15px;
                    margin: 10px 0;
                    font-size: 18px;
                    text-align: center;
                    box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
                ">
                    {e}
                </div>
                ''',
                unsafe_allow_html=True
            )
        st.session_state.errors = []
                    
    # --- History 버튼 ---
    st.markdown("---")
    st.subheader(f"History")

    model_history_dir = HISTORY_DIR / st.session_state.model_type
    if model_history_dir.exists():
        titles = sorted([p.name for p in model_history_dir.iterdir() if p.is_dir()])
        for t in titles:
            if st.button(t, key=f"history_{t}"):
                st.session_state.title = t
                st.session_state.page = "result"
                # 결과와 입력 로드
                input_path = model_history_dir / t / "input" / "input.txt"
                result_path = model_history_dir / t / "result" / "result.json"

                if input_path.exists():
                    with open(input_path, "r") as f:
                        st.session_state.text_input = f.read()
                if result_path.exists():
                    with open(result_path, "r") as f:
                        st.session_state.result = json.load(f)
                else:
                    st.session_state.result = None

                st.rerun()

###############################################
# PREVIEW PAGE
###############################################
def render_preview_page():
    
    st.markdown("<h1 style='text-align: center;'>Preview</h1>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown(f"<h3 style='text-align: left;'>Title: {st.session_state.title}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: left;'>Anomaly Type: {st.session_state.model_type}</h3>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown(f"<h2 style='text-align: center;'>Selected Option & User Input</h2>", unsafe_allow_html=True)
    
    preview_map = {
        "Image": preview_image,
        "Video": preview_video,
        "Log": preview_log,
        "Time": preview_time
    }
    
    preview_func = preview_map[st.session_state.model_type]
    preview_func()

    st.markdown("---")
    
    cols = st.columns([1]*9)
    with cols[0]:
        if st.session_state.input_error:
            proceed = st.button("Run", disabled=True)
        else:
            proceed = st.button("Run")
        
    with cols[8]:
        back = st.button("Back")
        
    if back:
        st.session_state.page = 'input'
        st.rerun()
        
    if proceed:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.title = f"{st.session_state.title}_{timestamp}"
        run_dir = HISTORY_DIR / st.session_state.model_type / st.session_state.title
        
        input_dir = run_dir / "input"
        result_dir = run_dir / "result"
        input_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
            
        if st.session_state.text_input:
            with open(input_dir / "input.txt", "w") as f:
                f.write(st.session_state.text_input)
                
        if st.session_state.uploaded_file:
            for file in st.session_state.uploaded_file:
                with open(input_dir / file.name, "wb") as f:
                    f.write(file.getbuffer())

        with st.spinner("Waiting for prediction..."):
            try:
                endpoint_map = {
                    "Image": "http://localhost:8000/predict/image",
                    "Video": "http://localhost:8000/predict/video",
                    "Log": "http://localhost:8000/predict/log",
                    "Time": "http://localhost:8000/predict/time"
                }
                endpoint = endpoint_map[st.session_state.model_type]
                
                data = {} 
                if st.session_state.text_input:
                    data["text"] = st.session_state.text_input
                if st.session_state.model_type == 'Log':
                    data['log_type'] = st.session_state.log_type
                if st.session_state.model_type == 'Time':
                    data['time_type'] = st.session_state.time_type
                data["title"] = st.session_state.title
                    
                payload = {"data": json.dumps(data)}

                response = requests.post(endpoint, data=payload)
                try:
                    result = response.json()
                except:
                    st.error("Server returned non-JSON response.")
                    st.stop()

                if response.status_code != 200:
                    # FastAPI가 보내준 에러 메시지를 Streamlit에서 보여줌
                    st.error(f"Error: {result.get('error', 'Unknown error')}")
                    st.session_state.result = result
                    st.session_state.page = "result"
                    st.rerun()

                st.session_state.result = result
                st.session_state.page = "result"
                st.rerun()

            except Exception as e:
                st.error(f"Failed to get prediction: {e}")

###############################################
# RESULT PAGE
###############################################
def render_result_page():
    model_type = st.session_state.model_type
    result = st.session_state.result
    
    st.markdown("<h1 style='text-align: center;'>{}</h1>".format(model_type + " Anomaly Detection Result"), unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>{}</h2>".format(st.session_state.title), unsafe_allow_html=True)

    if model_type == "Image":
        image_result(result)
    elif model_type == "Video":
        video_result(result)
    elif model_type == "Log":
        log_result(result)
    elif model_type == "Time":
        time_result(result)

    st.markdown("---")
    
    if st.button("Back"):
        st.session_state.page = "input"
        st.rerun()


###############################################
# PAGE ROUTER
###############################################
if st.session_state.page == "input":
    st.session_state.text_input = ""
    render_input_page()
elif st.session_state.page == "preview":
    render_preview_page()
else:
    render_result_page()