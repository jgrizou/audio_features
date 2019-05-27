import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

import shutil

import tools


source_folder = os.path.join(HERE_PATH, 'samples', 'rouanet')

destination_folder = os.path.join(HERE_PATH, 'samples', 'french')
tools.ensure_dir(destination_folder)

label_file = os.path.join(source_folder, 'wavToTag.txt')
with open(label_file, 'r') as f:
    labels = [l.strip() for l in f.readlines()]

for i in range(len(labels)):
    label = labels[i]
    target_folder = os.path.join(destination_folder, label)
    tools.ensure_dir(target_folder)

    n_files = len(tools.list_files(target_folder))
    target_file = os.path.join(target_folder, '{}.wav'.format(n_files))

    source_file = os.path.join(source_folder, '{}.wav'.format(i))

    shutil.copy(source_file, target_file)
