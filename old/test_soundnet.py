import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

import tools

import numpy as np

from dtw import dtw
from dtw import accelerated_dtw

import librosa
import librosa.display


SAMPLE_FOLDER = os.path.join(HERE_PATH, 'sound_samples')
TEST_FOLDER = os.path.join(HERE_PATH, 'sound_tests')


if __name__ == '__main__':

    import soundnet

    #Review of the model and architecture parameters
    model = soundnet.build_model()
    model.summary()


    from keras import backend as K
    #get Hidden Representation function
    get_layer_output = K.function([model.layers[0].input],
                                  [model.layers[28].output])

    # layer_output = get_layer_output([audio])[0] # multidimensional vector
    # tensor = layer_output.reshape(1,-1) # change vector shape to 1 (tensor)

    def pad_audio(audio):

        AIMED_LENGHT = 22050*5

        SCTRECH_LENGHT = 22050*5.5
        current_lenght = audio.shape[0]

        ratio_to_target = current_lenght / AIMED_LENGHT

        audio = librosa.effects.time_stretch(audio, ratio_to_target)

        # while audio.shape[0] < AIMED_LENGHT:
        #     audio = np.concatenate([audio, audio])

        return audio[:AIMED_LENGHT]


    def compute_features(audio_filepath):
        audio, sr = librosa.load(audio_filepath, dtype='float32', sr=22050, mono=True)
        audio = pad_audio(audio)
        audio = soundnet.preprocess(audio)
        layer_output = get_layer_output([audio])[0]
        activations = layer_output.reshape(1,-1)
        list_activations = activations.tolist()[0]
        return list_activations


    import tools

    tag_file = os.path.join(TEST_FOLDER, 'wavToTag.txt')
    with open(tag_file, 'r') as f:
        labels = [l.replace('\n', '') for l in f.readlines()]

    # labels = labels[:10]

    test_features = []
    for i in range(len(labels)):
        filepath = os.path.join(TEST_FOLDER, '{}.wav'.format(i))
        test_features.append(compute_features(filepath))


    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    y_true = le.fit_transform(labels)


    from sklearn.manifold import TSNE
    X = [np.array(t) for t in test_features]
    X = np.array(X)

    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # X_tsne = scaler.fit_transform(X)

    X_tsne = X

    tSNE_params = {
        'n_components': 2,
        'perplexity': 50,
        'n_iter': 5000,
        'n_iter_without_progress': 5000
    }

    tsne = TSNE(**tSNE_params)
    X_embedded = tsne.fit_transform(X_tsne)


    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    plt.ion()

    plt.close('all')

    plt.scatter(X_embedded[:,0], X_embedded[:,1], c=y_true)
