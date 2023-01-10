import os
import wave
import time
import pickle
import pyaudio
import warnings
import random
import numpy as np
from sklearn import preprocessing
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn.mixture import GaussianMixture

import time
from tkinter import *
from tkinter import messagebox
from itertools import count


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 512
DURATION_SECONDS = 5
NO_TEST_SAMPLES = 2
PASSWORD = list(open('passwords.txt'))
COUNTER = 0


def chooseRecordingDevice():
    pyAudio = pyaudio.PyAudio()
    list_of_devices = []
    #print('Wybierz mikrofon:')
    data = pyAudio.get_host_api_info_by_index(0)
    for i in range(0, data.get('deviceCount')):
        if (pyAudio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            list_of_devices.append(
                pyAudio.get_device_info_by_host_api_device_index(0, i).get('name'))
    #device = int(input())
    #print("Wybrane urządzenie: " + str(device))
    return list_of_devices


def record_sample():
    pyAudio = pyaudio.PyAudio()

    record_device = 1
    stream = pyAudio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=record_device,
        frames_per_buffer=CHUNK
    )

    label_record = Label(ws, text="Nagrywam...", font='Arial 15')
    label_record.place(relx=0.5, rely=0.25, anchor=CENTER)
    ws.update()
    frames = []
    for i in range(0, int(RATE / CHUNK * DURATION_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    label_record = Label(ws, text="Koniec nagrywania...", font='Arial 15')
    label_record.place(relx=0.5, rely=0.25, anchor=CENTER)
    ws.update()
    stream.stop_stream()
    stream.close()
    pyAudio.terminate()

    OUTPUT_FILENAME = "sample.wav"
    WAVE_OUTPUT_FILENAME = os.path.join("testing_set", OUTPUT_FILENAME)
    trainedfilelist = open("testing_set_addition.txt", 'a')
    trainedfilelist.write(OUTPUT_FILENAME+"\n")
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(pyAudio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    label_record = Label(ws, text="Identyfikacja w toku...", font='Arial 15')
    label_record.place(relx=0.5, rely=0.25, anchor=CENTER)
    ws.update()
    time.sleep(1)
    winner = "Zidentyfikowano: " + test_model()
    label_record = Label(ws, text=winner, font='Arial 15')
    label_record.place(relx=0.5, rely=0.25, anchor=CENTER)
    ws.update()


def calculate_delta(array):

    rows, cols = array.shape
    print(rows)
    print(cols)
    deltas = np.zeros((rows, 26))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
                first = 0
            else:
                first = i-j
            if i+j > rows-1:
                second = rows-1
            else:
                second = i+j
            index.append((second, first))
            j += 1
        deltas[i] = (array[index[0][0]]-array[index[0][1]] +
                     (2 * (array[index[1][0]]-array[index[1][1]]))) / 10
    return deltas


def extract_features(audio, rate):

    mfcc_feature = mfcc.mfcc(audio, rate, 0.025, 0.01,
                             39, nfft=1200, appendEnergy=True)
    mfcc_feature = preprocessing.scale(mfcc_feature)
    print(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature, delta))
    return combined


def train_model():
    source = "C:\\Users\\intel\\Downloads\\Speaker-Identification-Using-Machine-Learning-master\\Speaker-Identification-Using-Machine-Learning-master\\training_set\\"
    dest = "C:\\Users\\intel\\Downloads\\Speaker-Identification-Using-Machine-Learning-master\\Speaker-Identification-Using-Machine-Learning-master\\trained_models\\"
    train_file = "C:\\Users\\intel\\Downloads\\Speaker-Identification-Using-Machine-Learning-master\Speaker-Identification-Using-Machine-Learning-master\\training_set_addition.txt"
    file_paths = open(train_file, 'r')
    count = 1
    features = np.asarray(())
    for path in file_paths:
        path = path.strip()
        print(path)

        sr, audio = read(source + path)
        print(sr)
        vector = extract_features(audio, sr)

        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))

        if count == 5:
            gmm = GaussianMixture(
                n_components=6, max_iter=200, covariance_type='diag', n_init=3)
            gmm.fit(features)

            # dumping the trained gaussian model
            picklefile = path.split("-")[0]+".gmm"
            pickle.dump(gmm, open(dest + picklefile, 'wb'))
            print('+ modeling completed for speaker:', picklefile,
                  " with data point = ", features.shape)
            features = np.asarray(())
            count = 0
        count = count + 1


def test_model():

    source = "C:\\Users\\intel\\Downloads\\Speaker-Identification-Using-Machine-Learning-master\\Speaker-Identification-Using-Machine-Learning-master\\testing_set\\"
    modelpath = "C:\\Users\\intel\\Downloads\\Speaker-Identification-Using-Machine-Learning-master\\Speaker-Identification-Using-Machine-Learning-master\\trained_models\\"
    test_file = "C:\\Users\\intel\\Downloads\\Speaker-Identification-Using-Machine-Learning-master\\Speaker-Identification-Using-Machine-Learning-master\\testing_set_addition.txt"
    file_paths = open(test_file, 'r')

    gmm_files = [os.path.join(modelpath, fname) for fname in
                 os.listdir(modelpath) if fname.endswith('.gmm')]

    # Load the Gaussian gender Models
    models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
    speakers = [fname.split("\\")[-1].split(".gmm")[0] for fname
                in gmm_files]

    # Read the test directory and get the list of test audio files
    for path in file_paths:

        path = path.strip()
        print(path)
        sr, audio = read(source + path)
        vector = extract_features(audio, sr)

        log_likelihood = np.zeros(len(models))

        for i in range(len(models)):
            gmm = models[i]  # checking with each model one by one
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()

        winner = np.argmax(log_likelihood)
        print("\tdetected as - ", speakers[winner])
    # print(speakers[winner])
        return speakers[winner]


def show_password(ws):
    label = Label(ws, text=random.choice(PASSWORD), font='Arial 17 bold')
    label.place(relx=0.5, rely=0.6, anchor=CENTER)


def create_profile(wsnew, modelName, label_password, label_record):
    #modelName = modelName.get()
    # print(modelName)
    #modelName = (input("Nazwa użytkownika:"))

    pyAudio = pyaudio.PyAudio()
    recordingDevice = 1  # chooseRecordingDevice()
    global COUNTER
    # for count in range(NO_TEST_SAMPLES):
    stream = pyAudio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=recordingDevice,
        frames_per_buffer=CHUNK
    )

    label_password.place_forget()
    label_password = Label(wsnew, text=PASSWORD[COUNTER], font='Arial 17 bold')
    label_password.place(relx=0.5, rely=0.6, anchor=CENTER)

    label_record = Label(wsnew, text="Nagrywam...", font='Arial 15')
    label_record.place(relx=0.5, rely=0.25, anchor=CENTER)
    wsnew.update()
    frames = []
    for i in range(0, int(RATE / CHUNK * DURATION_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    label_record.place_forget()
    label_record = Label(wsnew, text="Koniec nagrania...", font='Arial 15')
    label_record.place(relx=0.5, rely=0.25, anchor=CENTER)
    stream.stop_stream()
    stream.close()
    pyAudio.terminate()

    OUTPUT_FILENAME = modelName + "-sample" + str(COUNTER) + ".wav"
    WAVE_OUTPUT_FILENAME = os.path.join("training_set", OUTPUT_FILENAME)
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(pyAudio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    #label_record=Label(wsnew, text="Kolejne nagranie, przygotuj się...", font='Arial 15')
    #label_record.place(relx=0.5, rely=0.25, anchor=CENTER)
    time.sleep(5)
    label_record.place_forget()
    label_password.place_forget()
    COUNTER += 1


def add_user():
    wsnew = Tk()
    wsnew.geometry("700x350")
    wsnew.title("Identyfikacja użytkownika")

    navigation(ws)

    label_title = Label(wsnew, text="Dodawanie profilu użytkownika", font=f)
    label_title.place(relx=0.5, rely=0.10, anchor=CENTER)

    modelName = StringVar()

    name_label = Label(wsnew, text='Nazwa użytkownika',
                       font=('calibre', 10, 'bold'))
    name_entry = Entry(wsnew, textvariable=modelName,
                       font=('calibre', 10, 'normal'))
    name_label.place(relx=0.4, rely=0.3, anchor=CENTER)
    name_entry.place(relx=0.6, rely=0.3, anchor=CENTER)

    label_info = Label(
        wsnew, text="Wciśnij przycisk i wypowiedź wyświetlone poniżej hasło:", font='Arial 15')
    label_info.place(relx=0.5, rely=0.4, anchor=CENTER)

    label_password = Label(wsnew, text=PASSWORD[0], font='Arial 17 bold')
    label_password.place(relx=0.5, rely=0.6, anchor=CENTER)

    # show_password(wsnew)

    recordingDevice = chooseRecordingDevice()
    v = IntVar()
    for j in range(len(recordingDevice)):
        Radiobutton(wsnew,
                    text=recordingDevice[j],
                    padx=20,
                    variable=v,
                    value=j,
                    command=selection).pack(side="bottom")

    v.set(0)

    try:
        start_btn = Button(
            wsnew,
            text='START',
            bd='5',
            command=lambda: create_profile(
                wsnew, name_entry.get(), label_password, label_info)
        )

        start_btn.place(relx=0.5, rely=0.7, anchor=CENTER)

    except:
        print("problem")


def train():
    train_model()


def train_test():
    test_model()

# navigation


def navigation(window_name):
    frame = Frame(window_name)
    frame.pack()
    mainmenu = Menu(frame)
    mainmenu.add_command(label="Dodaj dane", command=add_user)
    mainmenu.add_command(label="Trenuj model", command=train)
    mainmenu.add_command(
        label="Identifikuj nagrane próbki", command=train_test)
    mainmenu.add_command(label="Exit", command=window_name.destroy)
    window_name.config(menu=mainmenu)


def selection():
    return str(v.get())


def startCountdown():
    label_record = Label(ws, text="Nagrywam...", font='Arial 15')
    label_record.place(relx=0.45, rely=0.25, anchor=CENTER)
    ws.update()
    record_voice("identify")
    label_record.place_forget()
    label_record = Label(ws, text="Nagranie zakończone...", font='Arial 15')
    label_record.place(relx=0.45, rely=0.25, anchor=CENTER)
    ws.update()

    test_model()


f = ("Arial", 24)

ws = Tk()
ws.geometry("700x350")
ws.title("Identyfikacja użytkownika")

navigation(ws)

label_title = Label(ws, text="Identyfikacja użytkownika", font=f)
label_title.place(relx=0.5, rely=0.10, anchor=CENTER)

label_info = Label(
    ws, text="Wciśnij przycisk i wypowiedź wyświetlone poniżej hasło:", font='Arial 15')
label_info.place(relx=0.5, rely=0.4, anchor=CENTER)

show_password(ws)

recordingDevice = chooseRecordingDevice()
v = IntVar()
for j in range(len(recordingDevice)):
    Radiobutton(ws,
                text=recordingDevice[j],
                padx=20,
                variable=v,
                value=j,
                command=selection).pack(side="bottom")

v.set(0)

start_btn = Button(
    ws,
    text='START',
    bd='5',
    command=record_sample
)

start_btn.place(relx=0.5, rely=0.7, anchor=CENTER)


ws.mainloop()
