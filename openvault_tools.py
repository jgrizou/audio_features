from collections import deque

import umap
import librosa
import numpy as np

import audio_tools
import embedding_tools


class AudioVaultSignal(object):

    def __init__(self):
        self.audioembedder = embedding_tools.AudioEmbedder()
        self.embeddings = []

    def add_feedback_mp3(self, feedback_mp3):
        y, sample_rate = librosa.load(feedback_mp3)
        y, _ = librosa.effects.trim(y)
        y = audio_tools.repeat_audio_to_duration(y, 3, sample_rate)
        y = audio_tools.frame_audio(y, 1, 0.1, sample_rate)
        embedding = self.audioembedder.compute_embeddings_from_array(y)

        self.embeddings.append(embedding)

    def get_feedback_signals(self):

        X = np.array(self.embeddings)

        X_umap = X.reshape((-1, 128))

        n_neighbors = min(150, X_umap.shape[0] - 1)

        X_mapped = umap.UMAP( n_neighbors=n_neighbors,
                                n_components=2,
                                min_dist=0.1,
                                metric='cosine').fit_transform(X_umap)

        mapped_per_signal_shape = (X.shape[0], X.shape[1], X_mapped.shape[1])
        X_mapped_per_signal = X_mapped.reshape(mapped_per_signal_shape)

        X_out = np.mean(X_mapped_per_signal, axis=1)

        ##
        results = {}
        results['embeddings_per_mp3'] = X
        results['embeddings_for_umap'] = X_umap
        results['mapped_embeddings_from_umap'] = X_mapped
        results['mapped_embeddings_per_mp3'] = X_mapped_per_signal
        results['mean_mapped_embeddings_per_mp3'] = X_out

        return X_out.tolist(), results


class AudioVaultPlayer(object):

    def __init__(self, n_hypothesis, positive_feedback_mp3_list, negative_feedback_mp3_list, target_index=None):
        self.n_hypothesis = n_hypothesis
        self.positive_feedback_mp3_deque = deque(positive_feedback_mp3_list)
        self.negative_feedback_mp3_deque = deque(negative_feedback_mp3_list)
        self.update_target_index(target_index)

    def update_target_index(self, new_target_index=None):
        if new_target_index is None:  # set it randomly
            self.target_index = np.random.randint(self.n_hypothesis)
        else:
            self.target_index = new_target_index

    def get_feedback_mp3(self, flash_pattern):
        is_target_flashed = flash_pattern[self.target_index]
        if is_target_flashed:
            deque_to_sample = self.positive_feedback_mp3_deque
        else:
            deque_to_sample = self.negative_feedback_mp3_deque

        deque_to_sample.rotate()
        return deque_to_sample[0]
