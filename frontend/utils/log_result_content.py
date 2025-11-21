import streamlit as st
from PIL import Image
import json
from pathlib import Path
import os
import re
from .util import text_area_style, make_download_sort
import pandas as pd
import openpyxl



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
        
        # df = pd.read_csv(output_file['detailed_logs_csv'])
        # st.dataframe(df)
        
        # df = pd.read_excel(output_file['styled_logs_excel'])
        # st.dataframe(df)
        
        wb = openpyxl.load_workbook(output_file['styled_logs_excel'])
        ws = wb.active

        custom_css = """
        <style>
        .excel-scroll-container {
            max-height: 600px;
            overflow-y: auto;
            border-radius: 12px;
            border: 1px solid;
            margin-top: 10px;
        }
        table.excel-table {
            width: 100%;                 /* 화면 너비에 맞춰 늘리고 줄어듬 */
            border-collapse: collapse;
            font-family: "Source Sans", sans-serif;
            font-size: 14px;
        }

        table.excel-table th, table.excel-table td {
            padding: 8px 12px;
            border-bottom: 1px solid;
            text-align: center;
            white-space: normal;          /* !!! 줄바꿈 허용 !!! */
            overflow: hidden;             /* 텍스트 넘침 방지 */
            text-overflow: ellipsis;      /* 너무 길면 ... 처리 (옵션) */
        }
        table.excel-table th {
            font-weight: bold;
        }
        .index-col {
            font-weight: bold;
            opacity: 0.7;
        }
        table.excel-table tr:hover {
        }
        </style>
        """

        rows = list(ws.iter_rows(values_only=False))

        header_cells = rows[0]                 # 실제 엑셀 헤더
        data_rows = rows[1:]                   # 실제 데이터 행

        # --- TABLE 시작 ---
        html = '<div class="excel-scroll-container">'
        html += '<table class="excel-table">'

        # --- 헤더 (맨 앞에 index 추가) ---
        html += "<tr>"
        html += "<th class='index-col'>index</th>"

        for cell in header_cells:
            col_name = cell.value if cell.value is not None else ""
            html += f"<th>{col_name}</th>"
        html += "</tr>"

        # --- 데이터 행 ---
        for idx, row in enumerate(data_rows):
            html += "<tr>"

            # index 추가
            html += f"<td class='index-col'>{idx}</td>"

            for cell in row:

                val = cell.value if cell.value is not None else ""

                fill = cell.fill
                if fill and fill.fgColor and fill.fgColor.type == "rgb":
                    bg = fill.fgColor.rgb

                style = ""
                if bg:
                    if bg == "00FFCDD2":
                        bg = "FF9999"
                    style += f"background-color: #{bg};"


                html += f'<td style="{style}">{val}</td>'

            html += "</tr>"

        html += "</table></div>"

        st.markdown(custom_css + html, unsafe_allow_html=True)
        
        
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
        
        second_col = df.columns[1]

        # 조건부 스타일링 함수
        def highlight_abnormal(row):
            if "abnormal" in str(row[second_col]):
                return ["background-color: #FF9999"]*len(row)  # 해당 행 전체 색
            else:
                return [""]*len(row)

        # 스타일 적용
        st.dataframe(df.style.apply(highlight_abnormal, axis=1))

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