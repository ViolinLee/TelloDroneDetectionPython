import os
import wave
import pyaudio
import soundfile
from paddlespeech.cli.tts.infer import TTSExecutor
from paddlespeech.cli.asr.infer import ASRExecutor


class SpeechAgent:
    def __init__(self):
        self.tts = TTSExecutor()
        self.asr = ASRExecutor()
        self.p = pyaudio.PyAudio()
        self.chunk = 1024

    def speech_recognize(self, wav_file: os.PathLike) -> str:
        result = self.asr(audio_file=wav_file)
        return result

    def asr_listen(self):
        """监听可造成阻塞（可封装在threading的run中，但需注意工具实例是否能跨线程调用）"""
        out_path = 'listening.wav'
        # 待实现监听和写入
        text = self.speech_generate(out_path)
        return text

    def speech_generate(self, text: str) -> os.PathLike:
        assert isinstance(text, str) and len(text) > 0, 'Input Chinese text...'
        wav_file = self.tts(text=text)
        return wav_file

    def tts_speak(self, text):
        wav_file = self.speech_generate(text)
        wf = wave.open(wav_file, 'rb')

        # open stream based on the wave object which has been input.
        stream = self.p.open(format=self.p.get_format_from_width(wf.getsampwidth()),
                             channels=wf.getnchannels(),
                             rate=wf.getframerate(),
                             output=True)

        data = wf.readframes(self.chunk)
        while data != b'':
            stream.write(data)
            data = wf.readframes(self.chunk)

        # cleanup stuff.
        wf.close()
        stream.close()

    def __del__(self):
        self.p.terminate()
