import io
import logging

import streamlit as st
from streamlit import caching
from scipy.io.wavfile import write
import tensorflow as tf
import torch

from data.audio import Audio
from tts.factory import tts_ljspeech
from vocoding.melgan.generator import Generator

tf.get_logger().setLevel('ERROR')
logging.root.handlers = []
logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(funcName)s: %(message)s',
                    level=logging.DEBUG)


def audio(wav, sr=22050):
    bytes_wav = bytes()
    byte_io = io.BytesIO(bytes_wav)
    write(byte_io, sr, wav)
    st.audio(byte_io, format='audio/wav')


@st.cache(allow_output_mutation=True)
def get_vocoder(voc_type: str):
    if voc_type == 'melgan':
        dictionary = torch.hub.load_state_dict_from_url(
            'https://github.com/seungwonpark/melgan/releases/download/v0.3-alpha/nvidia_tacotron2_LJ11_epoch6400.pt',
            map_location='cpu')
        vocoder = Generator(80, num_layers=[3, 3, 3, 3])
        vocoder.load_state_dict(dictionary['model_g'])
        vocoder.eval()
    return vocoder


st.title('Text to Speech')
st.markdown(
    'Text to Speech with [TransformerTTS](https://github.com/cfrancesco/TransformerTTS) and [MelGAN](https://github.com/seungwonpark/melgan)')

input_text = st.text_area(label='Type in some text',
                          value='Hello there, my name is LJ, an open-source voice.\n'
                                'Not to brag, but I am a fairly popular open-source voice.\n'
                                'A voice with a character.')

#
vocoder_type = 'melgan'
if st.button('GriffinLim'):
    caching.clear_cache()
    vocoder_type = 'griffinlim'

if st.button('MelGAN'):
    caching.clear_cache()
    vocoder_type = 'melgan'

logging.info(input_text)
model = tts_ljspeech('95000')
audio_class = Audio.from_config(model.config)

out = model.predict(input_text)
mel = out['mel'].numpy().T
if vocoder_type == 'griffinlim':
    wav = audio_class.reconstruct_waveform(out['mel'].numpy().T)

else:
    vocoder = get_vocoder(vocoder_type)
    mel = torch.tensor([mel])
    if torch.cuda.is_available():
        vocoder = vocoder.cuda()
        mel = mel.cuda()
    
    with torch.no_grad():
        wav = vocoder.inference(mel).numpy()

audio(wav, sr=audio_class.sampling_rate)
