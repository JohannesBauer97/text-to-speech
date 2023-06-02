from TTS.api import TTS
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
import numpy as np
import sounddevice as sd
import sys, os

# Ausgabe umleiten ins nichts
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Ausgabe wieder in die Konsole leiten
def enablePrint():
    sys.stdout = sys.__stdout__

# Funktion für die Initialisierung des Synthesizers
def InitSynthesizer(usepretrained=False, useVocoder=True):
    if usepretrained:
        # Vortrainiertes Modell verwenden
        path = TTS.get_models_file_path()
        model_manager = ModelManager(path)
        model_path, config_path, model_item = model_manager.download_model("tts_models/de/thorsten/tacotron2-DDC")
        voc_path, voc_config_path, _ = model_manager.download_model(model_item["default_vocoder"])
    else:
        # Eigenes Trainiertes Modell verwenden
        model_path = "./Thorsten-DE-Model/model.pth"
        config_path = "./Thorsten-DE-Model/model_config.json"
        voc_path = "./Thorsten-DE-Model/vocoder.pth"
        voc_config_path = "./Thorsten-DE-Model/vocoder_config.json"

    if not useVocoder:
        voc_path = None
        voc_config_path = None

    # Objekt für Synthese erstellen
    syn = Synthesizer(
        tts_checkpoint=model_path,
        tts_config_path=config_path,
        vocoder_checkpoint=voc_path,
        vocoder_config=voc_config_path
    )

    return syn

# Funktion um Text in Numpy Wavarray umzuwandeln
def TextToSpeech(text, syn):
    blockPrint()
    wavearray = syn.tts(text)
    enablePrint()
    return wavearray

# Funktion um WavArray abzuspielen, Sample Rate muss mit dem des Modells übereinstimmen!!!
def PlayWav(wavarray):
    sd.play(wavarray, 22050)
    sd.wait()

# Testaufruf
if __name__ == '__main__':
    synCustom = InitSynthesizer()
    synPretrained = InitSynthesizer(True)
    synPretrainedNoVocder = InitSynthesizer(True, False)
    wavCustom = TextToSpeech("Hallo, wie geht es dir?", synCustom)
    wavPretrained = TextToSpeech("Hallo, wie geht es dir?", synPretrained)
    wavPretrainedNoVocoder = TextToSpeech("Hallo, wie geht es dir?", synPretrainedNoVocder)

    input("\n Synthese abgeschlossen, drücke eine Taste für Ausgabe des selbst trainierten Models")
    PlayWav(wavCustom)
    input("\n Drücke eine Taste für Ausgabe des vortrainierten Models")
    PlayWav(wavPretrained)
    input("\n Drücke eine Taste für Ausgabe des vortrainierten Models ohne Vocoder")
    PlayWav(wavPretrained)
