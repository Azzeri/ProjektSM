import pyaudio
import wave
import os

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 512
DURATION_SECONDS = 2
NO_TEST_SAMPLES = 2


def chooseRecordingDevice():
    pyAudio = pyaudio.PyAudio()
    print('Wybierz mikrofon:')
    data = pyAudio.get_host_api_info_by_index(0)
    for i in range(0, data.get('deviceCount')):
        if (pyAudio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Identyfikator urządzenia: ", i, " - ",
                  pyAudio.get_device_info_by_host_api_device_index(0, i).get('name'))
    device = int(input())
    print("Wybrane urządzenie: " + str(device))
    return device


def record_test_samples():
    modelName = (input("Nazwa użytkownika:"))
    pyAudio = pyaudio.PyAudio()
    recordingDevice = chooseRecordingDevice()
    for count in range(NO_TEST_SAMPLES):
        stream = pyAudio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=recordingDevice,
            frames_per_buffer=CHUNK
        )

        print('Nagrywam...')
        frames = []
        for i in range(0, int(RATE / CHUNK * DURATION_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("Koniec nagrywania...")
        stream.stop_stream()
        stream.close()
        pyAudio.terminate()

        OUTPUT_FILENAME = modelName + "-sample" + str(count) + ".wav"
        WAVE_OUTPUT_FILENAME = os.path.join("test_files", OUTPUT_FILENAME)
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(pyAudio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()


def menu():
    while True:
        action = (int(input(
            "\nCo chcesz zrobić?\n 1.Nagraj próbki treningowe \n 2.Trening \n 3.Nagraj próbkę testową \n 4.Identyfikuj\n"
        )))

        if (action == 1):
            record_test_samples()
        elif (action == 2):
            print('Trening')
        elif (action == 3):
            print('Nagraj próbkę testową')
        elif (action == 4):
            print('Identyfikuj')
        if (action > 4):
            exit()


menu()
