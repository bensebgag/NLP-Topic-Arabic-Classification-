import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent)) 
from src.config import DATA_DIR, CLASSES

def load_dataset():
    all_texts = []   
    all_labels = []  
    
    for class_name in CLASSES:
        path_to_class_dir = DATA_DIR / class_name
        
        if not path_to_class_dir.exists():
           print(f"Warning: Directory {path_to_class_dir} does not exist. Skipping.")
           continue
           
        txt_files = list(path_to_class_dir.glob("*.txt"))
        count = 0
        
        for file_path in txt_files:
            try:
                content = file_path.read_text(encoding='utf-8', errors='strict')
                if content and content.strip():
                    all_texts.append(content.strip())
                    all_labels.append(class_name)
                    count += 1
            except Exception as e:
                   print(f"Error reading file {file_path}: {e}")
                
        print(f"done loading {count} files for class '{class_name}'")
        
    df = pd.DataFrame({
        'text': all_texts,
        'label': all_labels
    })
    
    print(f"\nTotal articles loaded: {len(df)}")
    return df
