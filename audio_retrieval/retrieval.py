import numpy as np
import tqdm.auto as tqdm
from .model import FingerPrintModel, ModelConfig
import torch
from .preprocessing import load_mel_spectrogram, split_spectrogram
from .indexing import Indexer
from typing import List, Tuple
from collections import OrderedDict
import os
import warnings
import random

# Suppress OpenMP warnings and multiprocessing warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing.resource_tracker')

import torch
torch.set_num_threads(1)


class AudioRetrieval:
    def __init__(self, model_config: ModelConfig):
        self.model = self._prepare_model(model_config)
        self.device = model_config.device
        self.indexer = Indexer(dim=model_config.d_model)  # Assuming embedding size is 64
        self.segments = []
        self.labels = []

    def _prepare_model(self, model_config: ModelConfig) -> FingerPrintModel:
        model = FingerPrintModel(
            d_model=model_config.d_model,
            hidden_size=model_config.hidden_size,
            u_size=model_config.u_size
        ).to(model_config.device)
        if model_config.model_path:
            state_dict = torch.load(model_config.model_path, map_location=model_config.device)
            model.load_state_dict(state_dict)
        model.eval()
        if model_config.compile_mode and model_config.compile_mode != 'None':
            model = torch.compile(model, mode=model_config.compile_mode)
        return model

    def _embed(self, segments: torch.Tensor) -> np.ndarray:
        embeddings = []
        labels = []
        with torch.no_grad():
            for label, segment in tqdm.tqdm(segments, desc="Embedding generation..."):
                segment = torch.from_numpy(segment).to(self.device).unsqueeze(0)
                embedding = self.model(segment.unsqueeze(0)).squeeze(0)
                embeddings.append(embedding.cpu().numpy())
                labels.append(label)
        return np.array(embeddings), labels

    def _embed_segments(self, segments: torch.Tensor, verbose: bool = False) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for segment in tqdm.tqdm(segments, desc="Embedding generation...", disable=not verbose):
                segment = torch.from_numpy(segment).to(self.device).unsqueeze(0)
                embedding = self.model(segment.unsqueeze(0)).squeeze(0)
                embeddings.append(embedding.cpu().numpy())
        return np.array(embeddings)

    def add_songs(self, file_path, label):
        log_mel_S, sr, hop_length = load_mel_spectrogram(file_path)
        segments = split_spectrogram(log_mel_S, sr, hop_length)
        if label not in self.labels:
            self.labels.append(label)
        for j in range(segments.shape[-1]):
            self.segments.append((label, segments[:, :, j]))
        print(f"Added {len(self.segments)} segments for {len(self.labels)} classes.")

    def load_n_segments(self, file_path, n, start_id=None):
        log_mel_S, sr, hop_length = load_mel_spectrogram(file_path)
        segments = split_spectrogram(log_mel_S, sr, hop_length)
        true_seg = []
        for j in range(segments.shape[-1]):
            true_seg.append(segments[:, :, j])
        if start_id is None:
            start_id = random.randint(0, max(0, len(true_seg) - n))
        left = start_id
        right = start_id + n
        return true_seg[left:right+1]
    
    def query_segments(self, segments: List[np.ndarray], top_k=5) -> List[Tuple[str, float]]:
        embeddings = self._embed_segments(segments, verbose=False)
        result = self.indexer.query(embeddings, top_k=top_k)
        predicted_labels = {}
        for label, frequency_score in result:
            if label in predicted_labels:
                predicted_labels[label].append(frequency_score)
            else:
                predicted_labels[label] = [frequency_score]
        label_mean_scores = {}
        for label, scores in predicted_labels.items():
            label_mean_scores[label] = sum(scores) / len(scores)
        return max(label_mean_scores.items(), key=lambda x: x[1], default=("unknown", 0.0))

    def build_index(self):
        if not self.segments:
            raise ValueError("No segments to index. Please add songs first.")
        embeddings, labels = self._embed(self.segments)
        self.indexer.add(embeddings, labels)

    def query(self, file_path: str, top_k=5) -> List[Tuple[str, float]]:
        log_mel_S, sr, hop_length = load_mel_spectrogram(file_path)
        segments = split_spectrogram(log_mel_S, sr, hop_length)
        s = []
        for j in range(segments.shape[-1]):
            s.append(segments[:, :, j])
        embeddings = self._embed_segments(s)
        result = self.indexer.query(embeddings, top_k=top_k)
        predicted_labels = {}
        for label, score in result:
            if label in predicted_labels:
                predicted_labels[label] += 1
            else:
                predicted_labels[label] = 1
        # Normalize scores to probabilities
        total = len(result)
        if total > 0:
            predicted_labels = {k: v / total for k, v in predicted_labels.items()}
        return max(predicted_labels.items(), key=lambda x: x[1], default=("unknown", 0.0))
