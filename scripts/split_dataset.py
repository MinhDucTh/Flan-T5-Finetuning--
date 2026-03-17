import json
import random
import os

def split_dataset(input_file, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Chia dataset từ file JSON thành 3 tập: train, validation, và test.
    
    Args:
        input_file: Đường dẫn tới file JSON chứa toàn bộ dataset (ví dụ: training_data.json)
        output_dir: Thư mục lưu 3 file đầu ra
        train_ratio: Tỉ lệ tập train
        val_ratio: Tỉ lệ tập validation
        test_ratio: Tỉ lệ tập test
        seed: Hạt giống ngẫu nhiên để đảm bảo tính nhất quán (reproducibility)
    """
    random.seed(seed)
    
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    total_samples = len(data)
    print(f"Total samples found: {total_samples}")
    
    # Shuffle dữ liệu để chia đều
    random.shuffle(data)
    
    # Tính toán số lượng cho từng tập
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    # Phần còn lại cho test để tránh sai số làm tròn
    test_size = total_samples - train_size - val_size 
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]
    
    print(f"Data splits: Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    
    # Đảm bảo thư mục đầu ra tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.json")
    val_path = os.path.join(output_dir, "validation.json")
    test_path = os.path.join(output_dir, "test.json")
    
    # Lưu ra file
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
        
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
        
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
        
    print(f"Saved successfully to '{output_dir}':")
    print(f" - {train_path}")
    print(f" - {val_path}")
    print(f" - {test_path}")

if __name__ == "__main__":
    input_file = r"D:\Project\Flan-T5\data\training_data.json"
    output_dir = r"D:\Project\Flan-T5\data\splits"
    
    if os.path.exists(input_file):
        split_dataset(input_file, output_dir)
    else:
        print(f"Input file not found: {input_file}")
