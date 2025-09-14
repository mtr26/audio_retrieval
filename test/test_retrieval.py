import sys
import os
import random

import numpy as np
import torch

# Direct import now that audio_retrieval is a proper module
from audio_retrieval import AudioRetrieval, ModelConfig

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

TOP_K = 20

model_config = ModelConfig(
    d_model=64, 
    hidden_size=1024, 
    u_size=32, 
    device='cpu',
    compile_mode='default',
    batch_size=64,
    model_path=os.path.join(os.path.dirname(__file__), "model_64-2.pth"))
retrieval_system = AudioRetrieval(model_config=model_config)

train_dir = os.path.join(os.path.dirname(__file__), "animals")
test_dir = os.path.join(os.path.dirname(__file__), "test")

# Load files + generate classes
for label in os.listdir(train_dir):
    label_dir = os.path.join(train_dir, label)
    if os.path.isdir(label_dir):
        for file_name in os.listdir(label_dir):
            if file_name.endswith(('.wav', '.mp3', '.flac', '.m4a', '.aac')):
                file_path = os.path.join(label_dir, file_name)
                retrieval_system.add_songs(file_path, label)

retrieval_system.build_index()
results_1s = []
results_3s = []
results_10s = []
for actual_label in os.listdir(test_dir):
    test_label_dir = os.path.join(test_dir, actual_label)
    if not os.path.isdir(test_label_dir):
        continue
        
    for test_file in os.listdir(test_label_dir):
        if not test_file.endswith(('.wav', '.mp3', '.flac', '.m4a', '.aac')):
            continue
        test_path = os.path.join(test_label_dir, test_file)
        # One second clip
        print(f"Querying for {test_path} actural label: {actual_label}")
        segments_one_sec = retrieval_system.load_n_segments(test_path, n=1) # One second + 0.5 overlap
        predicted_label, confidence_score = retrieval_system.query_segments(segments_one_sec, top_k=TOP_K)
        results_1s.append(int(predicted_label == actual_label))
        print(f"Results for {test_file} (1 sec): predicted='{predicted_label}' (score={confidence_score:.3f}), actual='{actual_label}'")
        # Three second clip
        segments_three_sec = retrieval_system.load_n_segments(test_path, n=6) # Three seconds + 0.5 overlap
        predicted_label, confidence_score = retrieval_system.query_segments(segments_three_sec, top_k=TOP_K)
        results_3s.append(int(predicted_label == actual_label))
        print(f"Results for {test_file} (3 sec): predicted='{predicted_label}' (score={confidence_score:.3f}), actual='{actual_label}'")
        # Ten second clip
        segments_ten_sec = retrieval_system.load_n_segments(test_path, n=20) # Ten seconds + 0.5 overlap
        predicted_label, confidence_score = retrieval_system.query_segments(segments_ten_sec, top_k=TOP_K)
        results_10s.append(int(predicted_label == actual_label))
        print(f"Results for {test_file} (10 sec): predicted='{predicted_label}' (score={confidence_score:.3f}), actual='{actual_label}'")

print(f"Accuracy (1 sec): {sum(results_1s) / len(results_1s) * 100:.2f}%")
print(f"Accuracy (3 sec): {sum(results_3s) / len(results_3s) * 100:.2f}%")
print(f"Accuracy (10 sec): {sum(results_10s) / len(results_10s) * 100:.2f}%")
