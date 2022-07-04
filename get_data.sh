mkdir ./noise_data
cd noise_data
wget https://zenodo.org/record/2529934/files/FSDnoisy18k.audio_train.zip?download=1
wget https://zenodo.org/record/3670167/files/TAU-urban-acoustic-scenes-2020-mobile-development.audio.1.zip?download=1

unzip FSDnoisy18k.audio_train.zip?download=1
unzip TAU-urban-acoustic-scenes-2020-mobile-development.audio.1.zip?download=1

cd ..
mkdir ./clean_data
cd clean_data
wget https://www.openslr.org/resources/12/train-clean-360.tar.gz
tar -xf train-clean-360.tar.gz