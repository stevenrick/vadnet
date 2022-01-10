'''
ssivad.py
author: Johannes Wagner <wagner@hcm-lab.de>
created: 2018/05/04
Copyright (C) University of Augsburg, Lab for Human Centered Multimedia

Returns energy of a signal (dimensionwise or overall)
'''

import sys, os, json, argparse, glob

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pandas as pd
from scipy import stats
import librosa as lr
import soundfile as sf


def audio_from_file(path, sr=None, ext=''):
    return lr.load('{}{}'.format(path, ext), sr=sr, mono=True, offset=0.0, duration=None, dtype=np.float32, res_type='kaiser_best')                


def audio_to_file(path, x, sr):    
    sf.write(path, x.reshape(-1), sr, 'PCM_24')   


def audio_to_frames(x, n_frame, n_step=None):    

    if n_step is None:
        n_step = n_frame

    if len(x.shape) == 1:
        x.shape = (-1,1)

    n_overlap = n_frame - n_step
    n_frames = (x.shape[0] - n_overlap) // n_step       
    n_keep = n_frames * n_step + n_overlap

    strides = list(x.strides)
    strides[0] = strides[1] * n_step

    return np.lib.stride_tricks.as_strided(x[0:n_keep,:], (n_frames,n_frame), strides)

# 1032
def continuous_windows(x):
    gaps = []
    windows = []
    start = int(x[0])
    for i in range(len(x)):
        if i+1 < len(x):
            if x[i+1] - x[i] > 1:
                windows.append([start+0.0, int(x[i])+.99])
                gaps.append(int(x[i+1] - x[i]))
                start = int(x[i+1])
        if i == len(x):
           windows.append([start+0.0, int(x[i])+.99])
    # print(stats.describe(gaps))
    # print(windows)
    return windows

def extract_voice(path, files, n_batch=256, outDirectory=""):

    out_df = pd.DataFrame(columns=["File", "SpeechWindows"])

    print('load model from {}'.format(path))

    if os.path.isdir(path):
        candidates = glob.glob(os.path.join(path, 'model.ckpt-*.meta'))
        if candidates:
            candidates.sort()                
            checkpoint_path, _ = os.path.splitext(candidates[-1])
    else:
        checkpoint_path = path        

    if not all([os.path.exists(checkpoint_path + x) for x in ['.data-00000-of-00001', '.index', '.meta']]):
        print('ERROR: could not load model')
        raise FileNotFoundError

    vocabulary_path = checkpoint_path + '.json'
    if not os.path.exists(vocabulary_path):
        vocabulary_path = os.path.join(os.path.dirname(checkpoint_path), 'vocab.json')
    if not os.path.exists(vocabulary_path):
        print('ERROR: could not load vocabulary')
        raise FileNotFoundError

    with open(vocabulary_path, 'r') as fp:
        vocab = json.load(fp)

    graph = tf.Graph()

    with graph.as_default():

        saver = tf.train.import_meta_graph(checkpoint_path + '.meta')

        x = graph.get_tensor_by_name(vocab['x'])
        y = graph.get_tensor_by_name(vocab['y'])            
        init = graph.get_operation_by_name(vocab['init'])
        logits = graph.get_tensor_by_name(vocab['logits'])            
        ph_n_shuffle = graph.get_tensor_by_name(vocab['n_shuffle'])
        ph_n_repeat = graph.get_tensor_by_name(vocab['n_repeat'])
        ph_n_batch = graph.get_tensor_by_name(vocab['n_batch'])
        sr = vocab['sample_rate']

        with tf.Session() as sess:

            saver.restore(sess, checkpoint_path)
            idx = 0
            for file in files:

                print('processing {}'.format(file), flush=True)
                
                if os.path.exists(file):
                    sound, _ = audio_from_file(file, sr=sr)
                    input = audio_to_frames(sound, x.shape[1])
                    labels = np.zeros((input.shape[0],), dtype=np.int32)
                    sess.run(init, feed_dict = { x : input, y : labels, ph_n_shuffle : 1, ph_n_repeat : 1, ph_n_batch : n_batch })
                    count = 0
                    n_total = input.shape[0]
                    while True:
                        try:
                            output = sess.run(logits)
                            labels[count:count+output.shape[0]] = np.argmax(output, axis=1)
                            count += output.shape[0]
                            print('{:.2f}%\r'.format(100 * (count/n_total)), end='', flush=True)
                        except tf.errors.OutOfRangeError:
                            break
                    noise = input[np.argwhere(labels==0),:].reshape(-1,1)
                    speech = input[np.argwhere(labels==1),:].reshape(-1,1)

                    speech_segs = np.argwhere(labels==1)
                    if len(speech_segs) == 0:
                        print("no speech detected")
                        continue
                    # noise_segs = np.argwhere(labels==0)
                    speech_windows = continuous_windows(speech_segs)
                    # continuous_windows(noise_segs)

                    out_df.loc[idx] = [file.split(os.sep)[-1]] + [speech_windows]
                    idx += 1

                    if outDirectory != "":
                        outPath = os.path.join(outDirectory, os.path.basename(file))
                        name, ext = os.path.splitext(outPath)
                    else:
                        name, ext = os.path.splitext(file)
                    audio_to_file(os.path.join(name + '.speech' + ext), speech, sr)
                    # audio_to_file(os.path.join(name + '.noise' + ext), noise, sr)

                else:
                    print('skip [file not found]')
    if outDirectory != "":
        out_df.to_csv(os.path.join(outDirectory,"speech_segments.csv"), index=False)
    else:
        out_df.to_csv(os.path.join(directory,"speech_segments.csv"), index=False)


parser = argparse.ArgumentParser()

parser.add_argument('--model',
                default=r'models\vad',
                help='path to model')  

parser.add_argument('--files', 
                nargs='+', 
                default=[r'data\noise.wav', r'data\speech.wav'],
                help='list of files')

parser.add_argument('--n_batch', 
                type=int,
                default=256,
                help='number of batches')


if __name__ == '__main__':

    model = "models\\vad"
    directory = "D:\\steve\\Documents\\Downloads\\QUICK_VA_Audio"
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    batch_size = 256
    outDirectory = ""

    extract_voice(path=model, files=files)