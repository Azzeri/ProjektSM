import os
import wave
import time
import pickle
import pyaudio
import random
import numpy as np
from sklearn import preprocessing
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn.mixture import GaussianMixture
import time
from tkinter import *
from tkinter import messagebox

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 512
SAMPLE_DURATION_SECONDS = 5
NO_TEST_SAMPLES = 2
PASSWORDS = list(open('passwords.txt'))
LOCALPATH = "C:\\Users\\artes\\OneDrive\\Semestr 9\\SM\\Projekt\\"
TRAINING_MODELS = os.listdir(LOCALPATH + "trained_models\\")


class Identification:
    def __init__(self):
        self.ws_main = self.initialize_window("Identyfikacja")
        self.recording_device = IntVar()
        self.pyAudio = pyaudio.PyAudio()
        self.initialize_navigation()
        self.initialize_main_view()
        self.ws_main.mainloop()

    def initialize_window(self, title):
        ws = Tk()
        ws.geometry("700x350")
        ws.title(title)

        return ws

    def initialize_navigation(self):
        frame = Frame(self.ws_main)
        frame.pack()
        mainmenu = Menu(frame)
        mainmenu.add_command(label="Dodaj dane", command=self.nav_add_model)
        mainmenu.add_command(label="Trenuj modele",
                             command=self.nav_train_models)
        mainmenu.add_command(label="Exit", command=self.ws_main.destroy)
        self.ws_main.config(menu=mainmenu)

    def initialize_main_view(self):
        self.initialize_recording_devices()
        label_title = Label(
            self.ws_main, text="Identyfikacja użytkownika", font='Arial 18')
        label_title.place(relx=0.5, rely=0.10, anchor=CENTER)

        label_info = Label(
            self.ws_main, text="Wciśnij przycisk i wypowiedź wyświetlone poniżej hasło:", font='Arial 15')
        label_info.place(relx=0.5, rely=0.4, anchor=CENTER)

        label_password = Label(self.ws_main, text=random.choice(
            PASSWORDS), font='Arial 17 bold')
        label_password.place(relx=0.5, rely=0.6, anchor=CENTER)

        start_btn = Button(
            self.ws_main,
            text='START',
            bd='5',
            command=lambda: self.record_test_sample()
        )

        start_btn.place(relx=0.5, rely=0.7, anchor=CENTER)

    def initialize_recording_devices(self):
        recordingDevicesList = self.get_recording_devices()

        for i in range(len(recordingDevicesList)):
            Radiobutton(self.ws_main,
                        text=recordingDevicesList[i],
                        padx=20,
                        variable=self.recording_device,
                        value=i).pack(side="bottom")

    def get_recording_devices(self):
        pyAudio = pyaudio.PyAudio()
        list_of_devices = []
        data = pyAudio.get_host_api_info_by_index(0)
        for i in range(0, data.get('deviceCount')):
            if (pyAudio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                list_of_devices.append(
                    pyAudio.get_device_info_by_host_api_device_index(0, i).get('name'))

        return list_of_devices

    def calculate_delta(self, array):
        rows, cols = array.shape
        print(rows)
        print(cols)
        deltas = np.zeros((rows, 20))
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

    def extract_features(self, audio, rate):
        mfcc_feature = mfcc.mfcc(audio, rate, 0.025, 0.01,
                                 20, nfft=1200, appendEnergy=True)
        mfcc_feature = preprocessing.scale(mfcc_feature)
        print(mfcc_feature)
        delta = self.calculate_delta(mfcc_feature)
        combined = np.hstack((mfcc_feature, delta))
        return combined

    def add_model(self, ws_add_model, modelName, label_password, label_record):
        training_set = str(os.listdir(LOCALPATH + "training_set\\"))
        next_sample = (len([i for i in range(len(training_set))
                       if training_set.startswith(modelName, i)]))

        stream = self.pyAudio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=self.recording_device.get(),
            frames_per_buffer=CHUNK
        )

        label_password.place_forget()
        label_password = Label(
            ws_add_model, text=PASSWORDS[next_sample], font='Arial 17 bold')
        label_password.place(relx=0.5, rely=0.6, anchor=CENTER)

        label_record = Label(ws_add_model, text="Nagrywam...", font='Arial 15')
        label_record.place(relx=0.5, rely=0.25, anchor=CENTER)
        ws_add_model.update()

        frames = []
        for i in range(0, int(RATE / CHUNK * SAMPLE_DURATION_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        label_record.place_forget()
        label_record = Label(
            ws_add_model, text="Koniec nagrania...", font='Arial 15')
        label_record.place(relx=0.5, rely=0.25, anchor=CENTER)
        stream.stop_stream()
        stream.close()
        self.pyAudio.terminate()

        AUDIO_FILE_NAME = modelName + "-sample" + \
            str(next_sample) + ".wav"
        WAVE_OUTPUT_FILENAME = os.path.join("training_set", AUDIO_FILE_NAME)
        training_set_file = open("training_set_addition.txt", 'a')
        training_set_file.write("\n" + AUDIO_FILE_NAME)
        audio_file = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        audio_file.setnchannels(CHANNELS)
        audio_file.setsampwidth(self.pyAudio.get_sample_size(FORMAT))
        audio_file.setframerate(RATE)
        audio_file.writeframes(b''.join(frames))
        audio_file.close()

        time.sleep(1)
        label_record.place_forget()
        label_password.place_forget()

    def nav_add_model(self):
        ws_add_model = self.initialize_window("Nowy model")

        label_title = Label(
            ws_add_model, text="Dodawanie profilu użytkownika", font='Arial 18')
        label_title.place(relx=0.5, rely=0.10, anchor=CENTER)

        modelName = StringVar()

        name_label = Label(ws_add_model, text='Nazwa użytkownika',
                           font=('calibre', 10, 'bold'))
        name_entry = Entry(ws_add_model, textvariable=modelName,
                           font=('calibre', 10, 'normal'))
        name_label.place(relx=0.4, rely=0.3, anchor=CENTER)
        name_entry.place(relx=0.6, rely=0.3, anchor=CENTER)

        label_info = Label(
            ws_add_model, text="Wciśnij przycisk i wypowiedź wyświetlone poniżej hasło:", font='Arial 15')
        label_info.place(relx=0.5, rely=0.4, anchor=CENTER)

        label_password = Label(
            ws_add_model, text=PASSWORDS[0], font='Arial 17 bold')
        label_password.place(relx=0.5, rely=0.6, anchor=CENTER)

        start_btn = Button(
            ws_add_model,
            text='START',
            bd='5',
            command=lambda: self.add_model(
                ws_add_model, name_entry.get(), label_password, label_info)
        )

        start_btn.place(relx=0.5, rely=0.7, anchor=CENTER)

    def nav_train_models(self):
        ws_train_models = self.initialize_window("Trening")
        left_frame = Frame(ws_train_models)
        left_frame.grid(row=0, column=0)
        right_frame = Frame(ws_train_models)
        right_frame.grid(row=0, column=1)

        label_title = Label(
            ws_train_models, text="Trening", font='Arial 14'
        )
        label_title.place(relx=0.5, rely=0.10, anchor=CENTER)

        models_list = Listbox(left_frame)
        for index, file in enumerate(TRAINING_MODELS):
            models_list.insert(index, file.replace('.gmm', ''))
        models_list.pack()

        start_btn = Button(
            ws_train_models,
            text='START',
            bd='5',
            command=lambda: self.train_models(ws_train_models)
        )

        start_btn.place(relx=0.5, rely=0.7, anchor=CENTER)

    def train_models(self, ws_train_models):
        label_record = Label(
            ws_train_models, text="Nagrywam...", font='Arial 15')
        label_record.place(relx=0.5, rely=0.25, anchor=CENTER)
        ws_train_models.update()

        training_files_path = LOCALPATH + "training_set\\"
        trained_models_path = LOCALPATH + "trained_models\\"
        training_set_file = LOCALPATH + "training_set_addition.txt"
        file_paths = open(training_set_file, 'r')
        count = 1
        features = np.asarray(())

        for path in file_paths:
            path = path.strip()
            print(path)

            rate, audio = read(training_files_path + path)
            print(rate)
            vector = self.extract_features(audio, rate)

            if features.size == 0:
                features = vector
            else:
                features = np.vstack((features, vector))

            if count == 5:
                gmm = GaussianMixture(
                    n_components=6, max_iter=200, covariance_type='diag', n_init=3)
                gmm.fit(features)

                picklefile = path.split("-")[0]+".gmm"
                pickle.dump(gmm, open(trained_models_path + picklefile, 'wb'))
                print('+ modeling completed for speaker:', picklefile,
                      " with data point = ", features.shape)
                features = np.asarray(())
                count = 0
            count = count + 1

        label_record.place_forget()

    def record_test_sample(self):
        stream = self.pyAudio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=self.recording_device.get(),
            frames_per_buffer=CHUNK
        )

        label_record = Label(self.ws_main, text="Nagrywam...", font='Arial 15')
        label_record.place(relx=0.5, rely=0.25, anchor=CENTER)
        self.ws_main.update()

        frames = []
        for i in range(0, int(RATE / CHUNK * SAMPLE_DURATION_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        self.pyAudio.terminate()

        WAVE_OUTPUT_FILENAME = os.path.join("testing_set", "sample.wav")
        audio_file = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        audio_file.setnchannels(CHANNELS)
        audio_file.setsampwidth(self.pyAudio.get_sample_size(FORMAT))
        audio_file.setframerate(RATE)
        audio_file.writeframes(b''.join(frames))
        audio_file.close()

        self.identify_samples()

        label_record.place_forget()

    def identify_samples(self):
        source = LOCALPATH + "testing_set\\"
        trained_models_path = LOCALPATH + "trained_models\\"
        test_file = LOCALPATH + "testing_set_addition.txt"
        file_paths = open(test_file, 'r')

        gmm_files = [os.path.join(trained_models_path, fname) for fname in
                     os.listdir(trained_models_path) if fname.endswith('.gmm')]

        # Load the Gaussian gender Models
        models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
        speakers = [fname.split("\\")[-1].split(".gmm")[0] for fname
                    in gmm_files]

        # Read the test directory and get the list of test audio files
        for path in file_paths:

            path = path.strip()
            print(path)
            sr, audio = read(source + path)
            vector = self.extract_features(audio, sr)

            log_likelihood = np.zeros(len(models))

            for i in range(len(models)):
                gmm = models[i]  # checking with each model one by one
                scores = np.array(gmm.score(vector))
                log_likelihood[i] = scores.sum()

            winner = np.argmax(log_likelihood)
            print("\tdetected as - ", speakers[winner])
            messagebox.showinfo(
                "showinfo", f'Zidentyfikowano jako: {speakers[winner]}')
            time.sleep(1.0)


Identification()
