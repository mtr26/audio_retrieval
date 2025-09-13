import numpy as np
import librosa
import warnings
import math
warnings.filterwarnings('ignore')

def load_mel_spectrogram(file_path, sr=8000, n_fft=1024, hop_length=256, n_mels=256):
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=True)
    S_power = np.abs(D)**2
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_S = mel_basis.dot(S_power)
    log_mel_S = librosa.power_to_db(mel_S, ref=np.max)
    return log_mel_S, sr, hop_length

def split_spectrogram(log_mel_S, sr, hop_length, segment_duration_sec=1.0, overlap_sec=0.2, pad_mode='low'):
    # Compute frames per segment
    segment_frames = math.ceil(segment_duration_sec * sr / hop_length)
    hop_frames = int((segment_duration_sec - overlap_sec) * sr / hop_length)
    n_mels, n_frames = log_mel_S.shape

    # Optionally pad so that (n_frames - segment_frames) is divisible by hop_frames
    if pad_mode is not None:
        # Compute how many frames to pad
        if n_frames < segment_frames:
            pad_amount = segment_frames - n_frames
        else:
            remainder = (n_frames - segment_frames) % hop_frames
            pad_amount = (hop_frames - remainder) if remainder != 0 else 0

        if pad_amount > 0:
            # Choose a fill value: for log-mel in dB, use a low value, e.g., log_mel_S.min() or -80
            fill_value = log_mel_S.min() if pad_mode == 'low' else 0.0
            pad_array = np.full((n_mels, pad_amount), fill_value=fill_value, dtype=log_mel_S.dtype)
            log_mel_S = np.concatenate([log_mel_S, pad_array], axis=1)
            n_frames = log_mel_S.shape[1]

    segments = []
    # Loop over start indices
    for start in range(0, n_frames - segment_frames + 1, hop_frames):
        seg = log_mel_S[:, start:start + segment_frames]  # shape (n_mels, segment_frames)
        segments.append(seg)

    if segments:
        # Stack into array: shape (n_mels, segment_frames, n_segments)
        segments = np.stack(segments, axis=2)
    else:
        # No full segment fits
        segments = np.empty((n_mels, segment_frames, 0), dtype=log_mel_S.dtype)
    return segments
