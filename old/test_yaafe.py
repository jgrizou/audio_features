import numpy as np

from yaafelib import *

# Build a DataFlow object using FeaturePlan
fp = FeaturePlan(sample_rate=48000)
fp.addFeature('mfcc: MFCC blockSize=512 stepSize=256')
fp.addFeature('mfcc_d1: MFCC blockSize=512 stepSize=256 > Derivate DOrder=1')
fp.addFeature('mfcc_d2: MFCC blockSize=512 stepSize=256 > Derivate DOrder=2')
df = fp.getDataFlow()

# configure an Engine
engine = Engine()
engine.load(df)

# extract features from an audio file using AudioFileProcessor
afp = AudioFileProcessor()
audiofile = 'sound_samples/1.mp3'
afp.processFile(engine,audiofile)

feats = engine.readAllOutputs()
