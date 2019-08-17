#importing all necessary libraries
from std import *
from scipy.io import wavfile
from scipy import signal
import librosa

#training and testing csv files
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
labels = sorted(df_train.species.unique())

'''to change ['Ae. aegypti', 'Ae. albopictus', 'An. arabiensis', 'An. gambiae', 'C. pipiens', 'C. quinquefasciatus']
into respective numeric labels'''
def label_maker(label):
    for i,c in enumerate(labels):
        if c==label:
            return i

#this is the function to extract features from the sound files
def log_specgram(audio, sample_rate, window_size=20, step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, _, spec = signal.spectrogram(audio,
                                        fs = sample_rate,
                                        window = 'hann',
                                        nperseg = nperseg,
                                        noverlap = noverlap,
                                        detrend = False)

    return freqs, np.log(spec.T.astype(np.float32) + eps)

# this block of code will save the sound features as npy files for testing data
def create_test_npy():
    sound = []
    file_count = 0
    ct = 0
    n = len(df_train)
    for track in tqdm.tqdm(df_test['track_id']):
        folder = df_test[df_test['track_id'] == track].species.item()
        file = librosa.load('Wingbeats/'+folder+'/wav/'+track, sr=16000)
        _, data = log_specgram(file[0],file[1])
        sound.append((data, label_maker(folder)))
        ct+=1
        n-=1
        if ct==10000 or n==0:
            np.save('npy/test/'+str(file_count)+'.npy', sound)
            del sound[:]
            ct=0
            file_count+=1

#this function will save the npy for training data
def create_train_npy():
    sound = []
    file_count = 0
    ct = 0
    n = len(df_train)
    for track in tqdm.tqdm(df_train['track_id']):
        folder = df_test[df_train['track_id'] == track].species.item()
        file = librosa.load('Wingbeats/'+folder+'/wav/'+track, sr=16000)
        _, data = log_specgram(file[0],file[1])
        sound.append((data, label_maker(folder)))
        ct+=1
        n-=1
        if ct==10000 or n==0:
            np.save('npy/train/'+str(file_count)+'.npy', sound)
            del sound[:]
            ct=0
            file_count+=1
