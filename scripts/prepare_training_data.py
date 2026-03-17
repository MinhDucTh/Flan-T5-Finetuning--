import os
import json
import random

def generate_synthetic_questions(text):
    """
    Sinh 3 câu hỏi trắc nghiệm giả lập từ đoạn text để có dữ liệu huấn luyện.
    Vì text là hội thoại không có cấu trúc rõ ràng, chúng ta sẽ tạo các câu hỏi dựa trên
    các từ ngẫu nhiên có độ dài > 5 ký tự trong text để làm đáp án.
    """
    words = [w.strip(".,!?\"'") for w in text.split()]
    words = [w for w in words if len(w) > 5]
    
    if len(words) < 12:
        # Nếu ít từ quá, thêm vài từ dummy
        words += ["something", "happened", "morning", "afternoon", "evening"] * 3
        
    questions = []
    
    for i in range(1, 4):
        # Chọn 1 từ làm đáp án đúng
        answer_word = random.choice(words)
        # Chọn 3 từ khác làm đáp án sai
        wrong_words = random.sample([w for w in words if w != answer_word] + ["apple", "banana", "car", "dog", "house"], 3)
        
        options = [answer_word] + wrong_words
        random.shuffle(options)
        
        answer_idx = options.index(answer_word)
        answer_letter = chr(65 + answer_idx) # A, B, C, D
        
        q_text = f"Question {i}: Which word was mentioned in the conversation regarding the main topic?\n"
        q_text += f"A. {options[0]}\n"
        q_text += f"B. {options[1]}\n"
        q_text += f"C. {options[2]}\n"
        q_text += f"D. {options[3]}\n"
        q_text += f"Answer: {answer_letter}"
        
        questions.append(q_text)
        
    return "\n\n".join(questions)

def main():
    base_dir = r"D:\Project\Flan-T5\data\processed_data_old"
    output_file = r"D:\Project\Flan-T5\data\training_data.json"
    
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    videos = sorted(os.listdir(base_dir))
    dataset = []
    
    for vid in videos:
        vpath = os.path.join(base_dir, vid)
        if not os.path.isdir(vpath):
            continue
            
        chunk_dir = os.path.join(vpath, "chunks")
        if not os.path.isdir(chunk_dir):
            continue
            
        chunks = []
        chunk_files = sorted(os.listdir(chunk_dir))
        
        for cf in chunk_files:
            cfp = os.path.join(chunk_dir, cf)
            with open(cfp, "r", encoding="utf-8") as f:
                content = f.read().strip()
                # Bỏ qua các chunks quá nhỏ và rác
                if len(content) > 50:
                    chunks.append(content)
                elif len(content) > 0 and len(chunks) > 0:
                    # Nối vào chunk trước đó nếu là đoạn cắt dở
                    chunks[-1] += " " + content
                    
        if not chunks:
            continue
            
        # Nối tất cả text lại để sinh câu hỏi
        full_text = " ".join(chunks)
        reference_questions = generate_synthetic_questions(full_text)
        
        dataset.append({
            "video_id": vid,
            "chunks": chunks,
            "reference_questions": reference_questions
        })
        
    print(f"Processed {len(dataset)} videos.")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
        
    print(f"Saved generated training data to {output_file}")

if __name__ == "__main__":
    main()
