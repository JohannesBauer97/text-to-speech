#pip install tts

import os
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
import torch.multiprocessing as mp
import torch

# Windows bug bei Paralell Processing -> Kein Plan warum dass das Problem behebt... unter Linux/Mac scheinbar nicht notwendig...
if __name__ == '__main__':
    
    # GPU Sichtbar machen, sofern GPU vorhanden
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 0: # Haupt GPU verwenden!
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Ausgabepfad für Traingslauf
    output_path = "J:/TTS_NEW/train"
    
    # Definieren eines Datasets. ljspeech ist ein definiertes Format für Datasets
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", meta_file_train="metadata_train.csv", path="./"
    )
    
    # Konfiguration des TTS-Modells. Für Multithreading müssen die Anzahl Worker definiert werden, werden über CPU Cores ausgelesen
    # Konfiguration weitesgehend aus dem "Tutorial for beginners" übernommen, "epochs" von 1000 auf 100 geändert.
    cpu_cores = mp.cpu_count()
    config = GlowTTSConfig(
        batch_size=32,
        eval_batch_size=16,
        num_loader_workers=cpu_cores,
        num_eval_loader_workers=cpu_cores,
        precompute_num_workers=cpu_cores,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=100,
        text_cleaner="phoneme_cleaners",
        use_phonemes=True,
        phoneme_language="de-de",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        print_step=25,
        print_eval=False,
        mixed_precision=True,
        test_sentences=[
            "Es hat mich viel Zeit gekostet eine Stimme zu entwickeln, jetzt wo ich sie habe werde ich nicht mehr schweigen.",
            "Sei eine Stimme, kein Echo.",
            "Es tut mir Leid David. Das kann ich leider nicht machen.",
            "Dieser Kuchen ist großartig. Er ist so lecker und feucht.",
            "Vor dem 22. November 1963.",
        ],
        output_path=output_path,
        datasets=[dataset_config],
        save_step=1000,
    )
    
    # Audio processor extrahiert features und verarbeitet Audio-Files
    # SampleRate muss zu Trainingsdaten passen!
    ap = AudioProcessor.init_from_config(config)
    ap.sample_rate = 22050
    
    # Tokenizer für Umwandlung von Text in token IDs.
    tokenizer, config = TTSTokenizer.init_from_config(config)
    
    # Testdaten laden, TTS bietet eine Funktion für die automatische Ermittlung von Verhältnis Training-/Testdaten
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    
    # Modell intialisieren mit Config, AudioProcessor, Tokenizer un SpeakerManager
    # SpeakerManager ermöglicht ein TTS-Modell mit mehreren Sprachen oder Sprechern
    model = GlowTTS(config, ap, tokenizer, speaker_manager=None)
    
    # Trainer wird ebenfalls geliefert, Parameter werden hier zusammengeführt:
    # - Config
    # - Ausgabeordner
    # - Modell
    # - Training und Testdatensätze
    trainer = Trainer(
        TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )
  
    # Training starten
    trainer.fit()
