# FastAPI server
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import subprocess
import tempfile
from pathlib import Path
import json
import re


HISTORY_DIR = Path(__file__).parent.parent / "history"

app = FastAPI()

@app.post("/predict/image")
async def predict_image(text: str = Form(None)):
    # 실제 모델 예측
    result = {
        "prediction": "normal",
        "score": 0.92,
        "text_received": text if text else "No text",
        "file_name": file.filename if file else "No file"
    }
    return JSONResponse(content=result)

@app.post("/predict/video")
async def predict_video(text: str = Form(None)):
    # 실제 모델 예측
    result = {
        "prediction": "abnormal",
        "score": 0.70,
        "text_received": text if text else "No text",
        "file_name": file.filename if file else "No file"
    }
    return JSONResponse(content=result)

@app.post("/predict/log")
async def predict_log(data: str = Form(None)):
    
    parsed = json.loads(data) if data else {}
    
    text = parsed.get("text", "")
    title = parsed.get("title", "")
    log_type = parsed.get("log_type", "")
    
    pre_trained = ["BGL", "Thunderbird", "HDFS_v1"]
    
    if text or log_type not in pre_trained:
        bash_script = "log/log_unpretrained.sh"
    else:
        bash_script = "log/log_pretrained.sh"
    
    cmd = [
        "bash",
        bash_script,
        "--text", text,
        "--title", title,
        "--log_type", log_type
    ]
    
    result_json_path = HISTORY_DIR / 'Log' / title / "result" / "result.json"
    
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        
        if not result_json_path.exists():
            return JSONResponse(
                content={"error": "Something Wrong. Check Your Input Data"},
                status_code=500
            )
        
        with open(result_json_path) as f:
            result = json.load(f)

        return JSONResponse(content=result)
    
    except subprocess.CalledProcessError as e:
        # ValueError 메시지만 추출
        match = re.findall(r'ValueError:\s*(.+)', e.output)
        if match:
            error_msg = match[-1].strip()
        else:
            error_msg = e.output.strip()  # 혹시 매칭 실패하면 전체 출력

        return JSONResponse(
            content={
                "error": error_msg,
                "returncode": e.returncode,
                "cmd": e.cmd
            },
            status_code=500
        )


@app.post("/predict/time")
async def predict_time(data: str = Form(None)):

    parsed = json.loads(data) if data else {}
    
    text = parsed.get("text", "")
    title = parsed.get("title", "")
    method = parsed.get("time_type", "")
    
    if method == 'Method1':
        bash_script = "time/time_method1.sh"
    if method == 'Method2':
        bash_script = "time/time_method2.sh"
            
    cmd = [
        "bash",
        bash_script,
        "--text", text,
        "--title", title,
        "--method", method
    ]
    
    result_json_path = HISTORY_DIR / 'Time' / title / "result" / "result.json"
    
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        
        if not result_json_path.exists():
            return JSONResponse(
                content={"error": "Something Wrong. Check Your Input Data"},
                status_code=500
            )
        
        with open(result_json_path) as f:
            result = json.load(f)

        return JSONResponse(content=result)
    
    except subprocess.CalledProcessError as e:
        # ValueError 메시지만 추출
        match = re.findall(r'ValueError:\s*(.+)', e.output)
        if match:
            error_msg = match[-1].strip()
        else:
            error_msg = e.output.strip()  # 혹시 매칭 실패하면 전체 출력

        return JSONResponse(
            content={
                "error": error_msg,
                "returncode": e.returncode,
                "cmd": e.cmd
            },
            status_code=500
        )
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)