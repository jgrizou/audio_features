import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

import librosa
import numpy as np

import tools
import audio_tools
import embedding_tools

# adding openvault directory to path
import sys
openvault_path = os.path.join(HERE_PATH, '..')
sys.path.append(openvault_path)

from openvault.continuous import ContinuousLearner

from openvault_tools import AudioVaultPlayer


if __name__ == '__main__':

    type1_folder = os.path.join(HERE_PATH, 'samples', 'web_recordings', 'yellow')
    type1_files = tools.list_files(type1_folder, ['*.mp3'])

    type2_folder = os.path.join(HERE_PATH, 'samples', 'web_recordings', 'dog')
    type2_files = tools.list_files(type2_folder, ['*.mp3'])

    N_HYPOTHESIS = 10
    player = AudioVaultPlayer(N_HYPOTHESIS, type1_files, type2_files)
    learner = ContinuousLearner(N_HYPOTHESIS)


    for j in range(10):
        # flash_pattern = learner.get_next_flash_pattern(planning_method='even_random')
        flash_pattern = learner.get_next_flash_pattern(planning_method='even_uncertainty')

        feedback_mp3 = player.get_feedback_mp3(flash_pattern)

        print(feedback_mp3)

        # learner.update(flash_pattern, feedback_signal)
        # if learner.is_solved():
        #     pass


    audioembed = embedding_tools.AudioEmbedder()

    def compute_embedding(audio_path):
        y, sample_rate = librosa.load(audio_path)
        y, _ = librosa.effects.trim(y)
        y = audio_tools.repeat_audio_to_duration(y, 3, sample_rate)
        y = audio_tools.frame_audio(y, 1, 0.1, sample_rate)

        return audioembed.compute_embeddings_from_array(y)

    X = []
    y = []
    y_perm = []
    y_perm2 = []

    tools.set_seed(1)

    random.shuffle(type1_files)
    random.shuffle(type2_files)

    for t1 in type1_files[:10]:
        print(t1)
        E = compute_embedding(t1)
        X.extend(E)
        y.extend([0 for _ in E])

        y_random = np.random.randint(2)
        y_perm.extend([y_random for _ in E])

        y_random = np.random.randint(2)
        y_perm2.extend([y_random for _ in E])

    for t2 in type2_files[:10]:
        print(t2)
        E = compute_embedding(t2)
        X.extend(E)
        y.extend([1 for _ in E])

        y_random = np.random.randint(2)
        y_perm.extend([y_random for _ in E])

        y_random = np.random.randint(2)
        y_perm2.extend([y_random for _ in E])


    X = np.array(X)
    y = np.array(y)
    y_perm = np.array(y_perm)
    y_perm2 = np.array(y_perm2)

    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    plt.ion()

    plt.close('all')


    import umap

    X_umap = X
    X_embedded = umap.UMAP( n_neighbors=150,
                            n_components=2,
                            min_dist=0.1,
                            metric='cosine').fit_transform(X_umap)

    plt.figure(figsize=(15,5))
    ax = plt.subplot(131)
    plt.scatter(X_embedded[:,0], X_embedded[:,1], c=y)

    ax = plt.subplot(132)
    plt.scatter(X_embedded[:,0], X_embedded[:,1], c=y_perm)

    ax = plt.subplot(133)
    plt.scatter(X_embedded[:,0], X_embedded[:,1], c=y_perm2)

    ##

    # from scipy.spatial.distance import cdist
    #
    # metrics = ['euclidean', 'cosine', 'cityblock', 'seuclidean', 'correlation']
    # metrics = ['cosine']
    # for metric in metrics:
    #     Y = cdist(X, X, metric)
    #     fig, ax = plt.subplots()
    #     im = ax.imshow(Y)
    #     plt.title(metric)

    ##

    from sklearn.svm import SVC

    from sklearn.model_selection import cross_validate

    clf = SVC(gamma='auto')
    cv_results = cross_validate(clf, X_embedded, y, cv=3)
    print(cv_results['test_score'])

    cv_results = cross_validate(clf, X_embedded, y_perm, cv=3)
    print(cv_results['test_score'])

    cv_results = cross_validate(clf, X_embedded, y_perm2, cv=3)
    print(cv_results['test_score'])
