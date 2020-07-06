from pydub import AudioSegment
from pydub.playback import play

sound_path = "../sound/alarm.wav"
sound = AudioSegment.from_wav(sound_path)

def soundAlert(stop):
    while True:
        if stop():
            break
        print('playing sound')
        play(sound)

