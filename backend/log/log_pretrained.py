import argparse
import json
import os
from pathlib import Path
import re
import torch
from torch.utils.data import DataLoader
from model import LogLLM, LogLLM_not_ft
from customDataset import CustomDataset, CustomCollator
from utils import sliding_window_data, session_window_data, evalModel

HISTORY_DIR = Path(__file__).parent.parent.parent / "history"
ROOT_DIR = Path(__file__).parent

max_content_len = 100
max_seq_len = 128
batch_size = 32

Bert_path = "log/models/bert-base-uncased"
Llama_path = "log/models/Meta-Llama-3-8B"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--log_type", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda:0")

    input_save_path = HISTORY_DIR / 'Log' / args.title / "input"
    result_path = HISTORY_DIR / 'Log' / args.title / "result"
    
    input_text = args.text if args.text else ""
    
    samples_path = [p for p in input_save_path.iterdir() if p.is_file() and p.name != "input.txt"]
    
    ft_path = os.path.join(ROOT_DIR, r"ft_model_{}".format(args.log_type))
    
    result = {}
    result['text'] = []
    result['log_type'] = []
    result['input_paths'] = []
    # 삭제 필요할 수 있음
    result['output_paths'] = []
    
    for input_path in samples_path:
        
        if args.log_type == "HDFS":
            session_window_data(input_path, result_path)
        else:
            sliding_window_data(input_path, result_path, args.log_type)
            
        data_path = result_path / f'{Path(input_path).stem}_pred.csv'
        dataset = CustomDataset(data_path)
        
        
        model = LogLLM(Bert_path, Llama_path, ft_path=ft_path, is_train_mode=False, device=device,
                max_content_len=max_content_len, max_seq_len=max_seq_len)
        
        tokenizer = model.Bert_tokenizer
        collator = CustomCollator(tokenizer, max_seq_len=max_seq_len, max_content_len=max_content_len)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collator,
            num_workers=4,
            shuffle=False,
            drop_last=False
        )
            
        output_paths = evalModel(model, dataloader, result_path, Path(input_path).stem, device)
        
        result['text'].append(input_text)
        result['log_type'].append(args.log_type)
        result['input_paths'].append(str(input_path))
        result['output_paths'].append(output_paths)
        
    with open(result_path / "result.json", "w") as f:
        json.dump(result, f)
    

if __name__ == "__main__":
    main()