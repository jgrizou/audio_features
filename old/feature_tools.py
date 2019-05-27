import librosa


def compute_raw_features(sound):
    raw_features = {}

    raw_features['chroma_stft'] = librosa.feature.chroma_stft(sound)
    raw_features['chroma_cqt'] = librosa.feature.chroma_cqt(sound)
    raw_features['chroma_cens'] = librosa.feature.chroma_cens(sound)
    raw_features['mel'] = librosa.feature.melspectrogram(sound)
    raw_features['mfcc'] = librosa.feature.mfcc(sound, n_mfcc=40)
    raw_features['spectral_centroid'] = librosa.feature.spectral_centroid(sound)
    raw_features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(sound)
    raw_features['spectral_contrast'] = librosa.feature.spectral_contrast(sound)
    raw_features['spectral_flatness'] = librosa.feature.spectral_flatness(sound)
    raw_features['spectral_rolloff'] = librosa.feature.spectral_rolloff(sound)
    raw_features['tonnetz'] = librosa.feature.tonnetz(sound)
    raw_features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(sound)


    return raw_features
