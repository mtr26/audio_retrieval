# Audio Retrieval System

A neural audio fingerprinting system for content-based audio retrieval using deep learning. This project implements a convolutional neural network that can identify and classify audio content by learning robust audio fingerprints.

## 🎯 Overview

This system processes audio files by:
1. Converting audio to mel-spectrograms
2. Extracting segments using overlapping windows
3. Generating neural embeddings using a CNN-based fingerprint model
4. Building a searchable index using FAISS
5. Performing similarity search with cosine similarity

## 📁 Project Structure

```
audio_retrieval/
├── audio_retrieval/           # Main package
│   ├── __init__.py
│   ├── model.py              # Neural network architecture
│   ├── preprocessing.py      # Audio preprocessing utilities
│   ├── indexing.py          # FAISS-based similarity search
│   └── retrieval.py         # Main retrieval system
├── test/                    # Test data and evaluation
│   ├── test_retrieval.py    # Evaluation script
│   ├── animals/             # Training data
│   │   ├── birds/
│   │   ├── cats/
│   │   └── dogs/
│   └── test/               # Test data
│       ├── birds/
│       ├── cats/
│       └── dogs/
└── README.md
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch
- librosa
- FAISS
- numpy
- tqdm

### Install Dependencies
```bash
pip install torch librosa faiss-cpu numpy tqdm
```

## 🚀 Quick Start

### Basic Usage

```python
from audio_retrieval.retrieval import AudioRetrieval
from audio_retrieval.model import ModelConfig

# Configure the model
model_config = ModelConfig(
    d_model=64,           # Embedding dimension
    hidden_size=1024,     # Hidden layer size
    u_size=32,           # Projection layer size
    device='cpu'         # or 'cuda'
)

# Initialize retrieval system
retrieval_system = AudioRetrieval(model_config=model_config)

# Add audio files to the database
retrieval_system.add_songs('path/to/audio1.wav', 'label1')
retrieval_system.add_songs('path/to/audio2.wav', 'label2')

# Build the search index
retrieval_system.build_index()

# Query with an audio file
results = retrieval_system.query('path/to/query.wav', top_k=10)
print(f"Top match: {results[0]}")
```

### Advanced Usage: Segment-based Queries

```python
# Load specific number of segments for testing
segments = retrieval_system.load_n_segments('test_audio.wav', n=6)  # ~3 seconds
predicted_label, confidence = retrieval_system.query_segments(segments, top_k=20)
print(f"Prediction: {predicted_label} (confidence: {confidence:.3f})")
```

## 🧠 Model Architecture

### Neural Fingerprint Model
- **Encoder**: Convolutional layers for feature extraction from mel-spectrograms
- **Projection Head**: Maps features to normalized embedding space
- **Input**: 256×32 mel-spectrogram segments (256 mel bins, 32 time frames)
- **Output**: 64-dimensional L2-normalized embeddings

### Audio Processing Pipeline
1. **Sampling rate**: 8,000 Hz
2. **STFT**: 1024 window, 256 hop length
3. **Mel-spectrogram**: 256 mel bins, 32 time frames
4. **Fingerprint windows**: 1s with 0.5s overlap
5. **Dynamic range**: 80 dB
6. **Frequency range**: 300-4,000 Hz

## 📊 Evaluation

Run the evaluation script to test accuracy across different clip lengths:

```bash
cd test/
python test_retrieval.py
```

The script evaluates:
- **1-second clips**: Single segment queries
- **3-second clips**: 6-segment aggregated queries  
- **10-second clips**: 20-segment aggregated queries

### Example Output
```
Accuracy (1 sec): 45.67%
Accuracy (3 sec): 78.23%
Accuracy (10 sec): 89.45%
```

## 🔧 Configuration Options

### ModelConfig Parameters
```python
ModelConfig(
    d_model=64,              # Embedding dimension (64 or 128)
    hidden_size=1024,        # Hidden layer size
    u_size=32,              # Projection intermediate size
    device='cpu',           # Device: 'cpu' or 'cuda'
    model_path=None,        # Path to pre-trained weights
    compile_mode='default'  # PyTorch compilation mode
)
```

### Retrieval Parameters
- `top_k`: Number of similar segments to retrieve
- `n_segments`: Number of segments for multi-segment queries
- Overlap strategy: 0.5s overlap between 1s segments

## 📈 Performance Tips

1. **Batch Processing**: Use larger batch sizes for better GPU utilization
2. **Model Compilation**: Enable PyTorch compilation with `compile_mode='default'`
3. **FAISS Optimization**: Use GPU FAISS for large databases
4. **Memory Management**: Process long audio files in chunks

### Environment Variables
```bash
export KMP_DUPLICATE_LIB_OK=TRUE  # Fix OpenMP conflicts
export OMP_NUM_THREADS=1          # Limit threading
```

## Testing with Custom Data

### Prepare Your Dataset
```
your_data/
├── train/
│   ├── class1/
│   │   ├── audio1.wav
│   │   └── audio2.wav
│   └── class2/
│       ├── audio3.wav
│       └── audio4.wav
└── test/
    ├── class1/
    └── class2/
```

### Modify Test Script
```python
train_dir = "path/to/your_data/train"
test_dir = "path/to/your_data/test"
```

## 🔬 Technical Details

### Similarity Computation
- **Metric**: Cosine similarity (via FAISS IndexFlatIP)
- **Normalization**: L2-normalized embeddings
- **Aggregation**: Mean frequency scores across multiple segments

### Model Training
The model expects training with:
- Triplet loss or contrastive learning
- Data augmentation (pitch shift, time stretch, noise)
- Hard negative mining for better embeddings

## 📚 References

- Audio fingerprinting techniques and neural embeddings
- FAISS: A Library for Efficient Similarity Search
- Mel-spectrogram processing with librosa
- Contrastive learning for audio representation

## 📄 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here]

---

*For questions or issues, please check the troubleshooting section or open an issue.*
