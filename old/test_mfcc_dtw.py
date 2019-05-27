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

def compute_mfcc(filepath, n_mfcc=20):
    y, sr = librosa.load(filepath)
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)


if __name__ == '__main__':


    base_sound_files = tools.list_files(SAMPLE_FOLDER, ['*.mp3'])

    # base_sound_files = tools.list_files(TEST_FOLDER, ['*.wav'])[:20]

    base_mfccs = []
    for filepath in base_sound_files:
        base_mfccs.append(compute_mfcc(filepath))


    tag_file = os.path.join(TEST_FOLDER, 'wavToTag.txt')
    with open(tag_file, 'r') as f:
        labels = [l.replace('\n', '') for l in f.readlines()]

    # labels = labels[:10]

    test_mfccs = []
    for i in range(len(labels)):
        filepath = os.path.join(TEST_FOLDER, '{}.wav'.format(i))
        # print('{} = {}'.format(filepath, labels[i]))
        test_mfccs.append(compute_mfcc(filepath).tolist())

    test_features = []
    for i in range(len(labels)):
        sample_mfcc = np.array(test_mfccs[i])

        feature = []
        for base_mfcc in base_mfccs:
            dist, _, _, _ = accelerated_dtw(sample_mfcc.T, base_mfcc.T, 'cityblock')
            feature.append(dist)

        test_features.append(feature)


    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    y_true = le.fit_transform(labels)


    from sklearn.manifold import TSNE
    X = np.array(test_features)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_tsne = scaler.fit_transform(X)

    tSNE_params = {
        'n_components': 2,
        'perplexity': 30,
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



    #
    # fig = plt.figure(figsize=(4,6))
    #
    # ax1 = plt.subplot(2, 2, 1)
    # mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=20)
    # librosa.display.specshow(mfcc1)
    #
    # ax2 = plt.subplot(2, 2, 2)
    # mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=20)
    # librosa.display.specshow(mfcc2)
    #
    #
    # # from numpy.linalg import norm
    # # euclidean_norm = lambda x, y: norm(x - y, ord=2)
    # # dist, cost_matrix, acc_cost_matrix, path = dtw(mfcc1.T, mfcc2.T, dist=euclidean_norm)
    #
    # dist, cost_matrix, acc_cost_matrix, path = accelerated_dtw(mfcc1.T, mfcc2.T, 'euclidean')
    #
    # ax2 = plt.subplot(2, 1, 2)
    # cmap = plt.get_cmap('gray')
    #
    # plt.imshow(cost_matrix.T, origin='lower', cmap=cmap, interpolation='nearest')
    # plt.plot(path[0], path[1], 'w')
    # plt.xlim((-0.5, cost_matrix.shape[0]-0.5))
    # plt.ylim((-0.5, cost_matrix.shape[1]-0.5))
    #
    #
    # from sklearn.manifold import TSNE
    # X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    # X_embedded = TSNE(n_components=2).fit_transform(X)
    # X_embedded.shape
