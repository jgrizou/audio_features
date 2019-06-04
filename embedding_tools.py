import os

# this get our current location in the file system
import inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

import time

import json
import requests
import numpy as np

import docker

import uuid
import tempfile
import pysndfile


class AudioEmbedder(object):
    # https://github.com/IBM/MAX-Audio-Embedding-Generator

    def __init__(self):
        self.docker_port = 5001
        self.docker_container_name = 'AudioEmbedder'

        # check if container exists, if not create it
        try:
            docker.from_env().containers.get(self.docker_container_name)
        except docker.errors.NotFound:
            self.create_container()

        # make sure container is ready
        while not self.is_container_started():
            time.sleep(1)

    def create_container(self):
        self.docker_client.containers.run('codait/max-audio-embedding-generator', detach=True, ports={'5000/tcp': self.docker_port}, name=self.docker_container_name)

    def is_container_started(self):
        try:
            response = requests.get('http://127.0.0.1:{}/model/metadata'.format(self.docker_port))

            if response.ok:
                return True
            else:
                return Exception('Something wrong with container')

        except requests.ConnectionError:
            return False

    def compute_embeddings_from_file(self, audio_filepath):
        # This model recognizes a signed 16-bit PCM wav file as an input

        files = {
            'audio': (audio_filepath, open(audio_filepath, 'rb')),
        }

        response = requests.post('http://127.0.0.1:{}/model/predict'.format(self.docker_port), files=files)

        if response.ok:
            content = json.loads(response.content)
            return content['embedding']
        else:
            raise Exception('Something wrong in Embedding API call')

    def compute_embeddings_from_array(self, audio_array, sample_rate=22050):
        # save array to wav file and compute embedding on it
        # sample_rate=22050 is librosa default value
        # create a temporary directory using the context manager
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_audio_filepath = os.path.join(tmpdirname, '{}.wav'.format(uuid.uuid4()))
            pysndfile.sndio.write(tmp_audio_filepath, audio_array, rate=sample_rate, format='wav', enc='pcm16')
            embedding_vector = self.compute_embeddings_from_file(tmp_audio_filepath)
        # directory and contents have been removed
        return embedding_vector


if __name__ == '__main__':

    audio_filepath = os.path.join(HERE_PATH, 'tmp_pcm16.wav')

    audioembed = AudioEmbedder()

    import librosa

    y, _ = librosa.load(audio_filepath)

    X = audioembed.compute_embeddings_from_array(y)
