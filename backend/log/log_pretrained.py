import argparse
import json
import os
from pathlib import Path

def is_image(file_path):
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    return Path(file_path).suffix.lower() in image_exts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--title", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_dir)
    
    samples_path = os.listdir(input_path)
    
    for s in samples_path:
        if is_image(s):
            
        else:
            
    
    
    
    

    # 예시 inference
    result = {
        "status": "success",
        "input_path": str(input_path),
        "option": args.option,
        "prediction": "anomaly",
        "score": 0.78
    }
    
    # 결과 JSON 저장
    output_path = input_path.parent / "result.json"
    with open(output_path, "w") as f:
        json.dump(result, f)

    print("[log.py] Finished")
    print("[log.py] Result stored at:", output_path)

if __name__ == "__main__":
    main()