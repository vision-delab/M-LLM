import streamlit as st
from PIL import Image
import json
from pathlib import Path
import os
import re
from .util import text_area_style, make_download_sort
import pandas as pd


HISTORY_DIR = Path(__file__).parent.parent.parent / "history" / "Log"


def log_pre(result: dict):
    texts = result.get("text", [])
    log_type = result.get("log_type", [])
    input_paths = result.get("input_paths", [])
    output_paths = result.get("output_paths", [])
    
    for i in range(len(input_paths)):
        st.markdown("---")
    
        file_path = input_paths[i]
        st.subheader(f"Input File name: {Path(file_path).name}")
        make_download_sort(file_path, Path(file_path).name)
        
        output_file = output_paths[i]
        make_download_sort(output_file['detailed_logs_csv'], 'detailed_logs_csv')
        make_download_sort(output_file['styled_logs_excel'], 'styled_logs_excel')
        make_download_sort(output_file['integrated_report_excel'], 'integrated_report_excel')
        make_download_sort(output_file['hotspot_plot'], 'hotspot_plot')
        make_download_sort(output_file['summary_pie_chart'], 'summary_pie_chart')
        
        df = pd.read_csv(output_file['detailed_logs_csv'])
        st.dataframe(df)
        
        img = Image.open(output_file['hotspot_plot'])
        st.image(img, caption=f"Hotspot for detected Anomaly for {Path(file_path).name}")
    
        img = Image.open(output_file['summary_pie_chart'])
        st.image(img, caption=f"Pie chart for {Path(file_path).name}")


def log_unpre(result: dict):
    
    texts = result.get("text", [])
    log_type = result.get("log_type", [])
    input_paths = result.get("input_paths", [])
    detection_results = result.get("detection_results", [])
    final_file = result.get("final_file_path", [])

    # 입력 다운로드 
    low_input_paths = HISTORY_DIR / st.session_state.title
    
    for i in range(len(input_paths)):
        st.markdown("---")
        
        file_path = input_paths[i]
        st.subheader(f"Input File name: {Path(file_path).name}")
        
        make_download_sort(input_paths[i], Path(input_paths[i]).name)
        make_download_sort(final_file[i], Path(final_file[i]).name)
        
        if texts[i]:
            st.markdown(f"<h3 style='text-align: left;'>User Text</h3>", unsafe_allow_html=True)
            # text_area_style(texts[i])
            st.text_area("Input Text", texts[i], label_visibility="collapsed", key=f"text_{i}")
        else:
            st.markdown(f"<h3 style='text-align: left;'>No Input Text</h3>", unsafe_allow_html=True)
            
        with open(final_file[i], "r", encoding="utf-8") as f:
            data = json.load(f)    
        df = pd.DataFrame(data)
        st.dataframe(df)

        # with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        #     content = f.read()
        # numbered = "\n".join([
        #     f"{i+1}. {line}"
        #     for i, line in enumerate(content.splitlines())
        # ])
        # # text_area_style(numbered)
        # st.text_area("File Content", numbered, height=400, label_visibility="collapsed", key=f"File_{i}")
        
        # # prediction 보여주기
        # merged = []
        # for chunk in detection_results[i]:  
        #     lines = chunk.split("\n")
        #     merged.extend(lines)

        # # 기존 번호 제거 → "1. abnormal" → "abnormal"
        # cleaned = [re.sub(r"^\s*\d+\.\s*", "", line) for line in merged]
        
        # # 새 번호 다시 붙이기
        # renumbered = "\n".join([f"{i+1}. {line}" for i, line in enumerate(cleaned)])
        # # print(renumbered)
        
        # st.write("### Detection Results")
        # # text_area_style(renumbered)
        # st.text_area("Detections", renumbered, height=400, label_visibility="collapsed", key=f"predict_{i}")