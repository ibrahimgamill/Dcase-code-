import os
import numpy as np
import librosa
import scipy.io.wavfile as wavfile
import shutil
from tqdm import tqdm

def time_masking(audio, mask_factor):
    masked_audio = audio.copy()
    # Randomly select mask length
    mask_length = np.random.randint(1, mask_factor)
    start = np.random.randint(0, len(audio) - mask_length)
    # Replace masked portion with random noise
    masked_audio[start:start+mask_length] = np.random.normal(0, 0.1, mask_length)  
    return masked_audio

def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr

def save_audio(file_path, audio, sr):

    wavfile.write(file_path, sr, audio)

def augment_audio(input_dir, output_dir, min_mask_factor=5, max_mask_factor=15):
 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("============== BEGIN AUGMENTATION ==============")
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(".wav"):
            input_file_path = os.path.join(input_dir, filename)

            # Load audio
            audio, sr = load_audio(input_file_path)

            # Randomly select mask factor
            mask_factor = np.random.randint(min_mask_factor, max_mask_factor)

            # Apply time masking augmentation
            augmented_audio = time_masking(audio, mask_factor)

            # Generate output file path
            output_file_path = os.path.join(output_dir, filename.replace("_normal_", "_anomaly_"))

            # Save augmented audio
            save_audio(output_file_path, augmented_audio, sr)

            # Copy original file to output directory
            shutil.copy2(input_file_path, output_dir)

    print("============== END OF AUGMENTATION ==============")

def main(mask_factor=10):
    try:
        shutil.rmtree(os.path.join(os.getcwd(),'data','dcase2023t2','dev_data','raw','slider','train'))
        shutil.rmtree(os.path.join(os.getcwd(),'data','dcase2023t2','dev_data','processed','slider'))
    except FileNotFoundError:
        pass
    os.makedirs(os.path.join(os.getcwd(),'data','dcase2023t2','dev_data','raw','slider','train'))
    input_directory = os.path.join(os.getcwd(),'data','dcase2023t2','dev_data','raw','slider','normal')
    output_directory = os.path.join(os.getcwd(),'data','dcase2023t2','dev_data','raw','slider','train')
    
    augment_audio(input_directory, output_directory, mask_factor)

if __name__ == "__main__":
    main()
