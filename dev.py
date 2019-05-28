import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

import random
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
from openvault_tools import AudioVaultSignal

if __name__ == '__main__':

    tools.set_seed(10)

    mp3_root_folder = os.path.join(HERE_PATH, 'samples', 'web_recordings')
    mp3_folders = tools.list_folders(mp3_root_folder)

    for i in range(10):

        type1_folder = random.choice(mp3_folders)

        type2_folder = random.choice(mp3_folders)
        while type2_folder == type1_folder:
            type2_folder = random.choice(mp3_folders)


        type1_files = tools.list_files(type1_folder, ['*.mp3'])
        type2_files = tools.list_files(type2_folder, ['*.mp3'])

        N_HYPOTHESIS = 10
        player = AudioVaultPlayer(N_HYPOTHESIS, type1_files, type2_files)
        player_signal = AudioVaultSignal()
        learner = ContinuousLearner(N_HYPOTHESIS, proba_decision_threshold=0.95, proba_assigned_to_label_valid=0.95)

        solved = []
        correct = []
        for j in range(50):
            # flash_pattern = learner.get_next_flash_pattern(planning_method='even_random')
            flash_pattern = learner.get_next_flash_pattern(planning_method='even_uncertainty')

            feedback_mp3 = player.get_feedback_mp3(flash_pattern)
            # print(feedback_mp3)

            player_signal.add_feedback_mp3(feedback_mp3)

            feedback_signals, results = player_signal.get_feedback_signals()

            learner.signal_history = feedback_signals[:-1]
            learner.update(flash_pattern, feedback_signals[-1])

            #
            valid = False
            solved.append(learner.is_solved())
            if learner.is_solved():
                true_i_target = player.target_index
                found_i_target = learner.get_solution_index()
                print('{} - {} in {} steps'.format(true_i_target, found_i_target, j+1))
                valid = true_i_target == found_i_target

                # change target and propagate label for next target
                learner.propagate_labels_from_hypothesis(found_i_target)
                player.update_target_index()

            correct.append(valid)


    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    plt.ion()

    plt.close('all')

    plt.figure()
    plt.plot(learner.hypothesis_probability_history)

    plt.figure()
    plt.plot(solved)
    plt.plot(correct)

    X_scatter = np.array(feedback_signals)
    X_scatter_emb = np.array(results['mapped_embeddings_from_umap'])

    plt.figure(figsize=(15, 6))
    for i in range(10):
        ax = plt.subplot(2, 5, i+1)
        plt.scatter(X_scatter[:,0], X_scatter[:,1], c=learner.hypothesis_labels[i])
        if i == player.target_index:
            plt.title('*')

    plt.figure(figsize=(6, 6))
    plt.scatter(X_scatter_emb[:,0], X_scatter_emb[:,1], c='blue')
    plt.scatter(X_scatter[:,0], X_scatter[:,1], c='red')
