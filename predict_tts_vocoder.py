from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from data.audio import Audio
from vocoding.predictors import MelGANPredictor, HiFiGANPredictor
from tts.models import ForwardTransformer
from tts.factory import tts_ljspeech

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', '-p', dest='path')
    parser.add_argument('--step', dest='step', default='90000', type=str)
    parser.add_argument('--text', '-t', dest='text', default=None, type=str)
    parser.add_argument('--file', '-f', dest='file', default=None, type=str)
    parser.add_argument('--checkpoint', '-ckpt', dest='checkpoint', default=None, type=str)
    parser.add_argument('--outdir', '-o', dest='outdir', default=None, type=str)
    parser.add_argument('--store_mel', '-m', dest='store_mel', action='store_true')
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true')
    parser.add_argument('--all_weights', '-ww', dest='all_weights', action='store_true')
    parser.add_argument('--vocoder', dest='vocoder', default='glim')
    parser.add_argument('--voc_config', dest='voc_config', default='en')
    parser.add_argument('--single', '-s', dest='single', action='store_true')
    args = parser.parse_args()
    
    if args.file is not None:
        with open(args.file, 'r') as file:
            text = file.readlines()
        fname = Path(args.file).stem
    elif args.text is not None:
        text = [args.text]
        fname = 'custom_text'
    else:
        fname = None
        text = None
        print(f'Specify either an input text (-t "some text") or a text input file (-f /path/to/file.txt)')
        exit()

    if args.vocoder == 'glim':
        vocoder = None
    elif args.vocoder == 'melgan':
        vocoder = MelGANPredictor.from_folder(f'vocoding/melgan/{args.voc_config}/')
    elif args.vocoder == 'hifigan':
        vocoder = HiFiGANPredictor.from_folder(f'vocoding/hifigan/{args.voc_config}/')
    else:
        print('Error: vocoder must be either "glim", "melgan" or "hifigan"')
        exit()
    outdir = Path(args.outdir) if args.outdir is not None else Path('.')
    if args.path is not None:
        print(f'Loading model from {args.path}')
        model = ForwardTransformer.load_model(args.path)
    else:
        model = tts_ljspeech(args.step)
    file_name = f"{fname}_{model.config['data_name']}_{args.vocoder}_{model.config['git_hash']}_{model.config['step']}"

    outdir = outdir / 'outputs' / f'{fname}'
    outdir.mkdir(exist_ok=True, parents=True)
    output_path = (outdir / file_name).with_suffix('.wav')
    audio = Audio.from_config(model.config)
    print(f'Output wav under {output_path.parent}')
    wavs = []
    for i, text_line in enumerate(text):
        phons = model.text_pipeline.phonemizer(text_line)
        tokens = model.text_pipeline.tokenizer(phons)
        if args.verbose:
            print(f'Predicting {text_line}')
            print(f'Phonemes: "{phons}"')
            print(f'Tokens: "{tokens}"')
        out = model.predict(tokens, encode=False, phoneme_max_duration=None)
        mel = out['mel'].numpy().T
        if not vocoder:
            wav = audio.reconstruct_waveform(mel)
        else:
            wav = vocoder([mel])[0]
        wavs.append(wav)
        if args.store_mel:
            np.save((outdir / (file_name + f'_{i}')).with_suffix('.mel'), out['mel'].numpy())
        if args.single:
            audio.save_wav(wav, (outdir / (file_name + f'_{i}')).with_suffix('.wav'))
    audio.save_wav(np.concatenate(wavs), (outdir / file_name).with_suffix('.wav'))