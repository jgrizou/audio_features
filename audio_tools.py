import librosa
import numpy as np


def repeat_audio_to_duration(audio_array, target_duration_in_second, sample_rate):

    current_lenght = audio_array.shape[0]
    target_lenght = target_duration_in_second * sample_rate

    n_tiling = np.int(np.ceil(target_lenght / current_lenght))

    repeat_audio_array = np.tile(audio_array, n_tiling)[:target_lenght]

    assert repeat_audio_array.shape[0] == target_lenght

    return repeat_audio_array


def frame_audio(audio_array, window_lenght_second, hop_lenght_second, sample_rate):

    window_length = int(window_lenght_second * sample_rate)
    hop_length = int(hop_lenght_second * sample_rate)

    framed_audio = librosa.util.frame(audio_array, frame_length=window_length, hop_length=hop_length)

    # reshape to a single audio track
    return framed_audio.T.reshape((-1, ))
