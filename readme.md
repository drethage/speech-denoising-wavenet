A Wavenet For Speech Denoising
====

A neural network for end-to-end speech denoising, as described in: "[A Wavenet For Speech Denoising](https://arxiv.org/abs/1706.07162)"

Listen to denoised samples under varying noise conditions and SNRs [here](http://www.jordipons.me/apps/speech-denoising-wavenet/)

Installation
-----

It is recommended to use a [virtual environment](http://virtualenvwrapper.readthedocs.io/en/latest/install.html)

1. `git clone https://github.com/drethage/speech-denoising-wavenet.git`
2. `pip install -r requirements.txt`
3. Install [pygpu](http://deeplearning.net/software/libgpuarray/installation.html)

*Currently the project requires **Keras 1.2** and **Theano 0.9.0**, the large dilations present in the architecture are not supported by the current version of Tensorflow (1.2.0)*

Usage
-----

A pre-trained model (best-performing model described in the paper) can be found in `sessions/001/models` and is ready to be used out-of-the-box. The parameterization of this model is specified in `sessions/001/config.json`

*Download the dataset as described [below](https://github.com/drethage/speech-denoising-wavenet#dataset)*

#### Denoising:

Example: `THEANO_FLAGS=optimizer=fast_compile,device=gpu python main.py --mode inference --config sessions/001/config.json --noisy_input_path data/NSDTSEA/noisy_testset_wav --clean_input_path data/NSDTSEA/clean_testset_wav`

###### Speedup
To achieve faster denoising, one can increase the target-field length by use of the optional `--target_field_length` argument. This defines the amount of samples that are denoised in a single forward propagation, saving redundant calculations. In the following example, it is increased 10x that of when the model was trained, the batch_size is reduced to 4.

Faster Example: `THEANO_FLAGS=device=gpu python main.py --mode inference --target_field_length 16001 --batch_size 4 --config sessions/001/config.json --noisy_input_path data/NSDTSEA/noisy_testset_wav --clean_input_path data/NSDTSEA/clean_testset_wav`

#### Training:

`THEANO_FLAGS=device=gpu python main.py --mode training --config config.json`

#### Configuration
A detailed description of all configurable parameters can be found in [config.md](https://github.com/drethage/speech-denoising-wavenet/blob/master/config.md)

#### Optional command-line arguments:
Argument | Valid Inputs | Default | Description
-------- | ---- | ------- | -----
mode | [training, inference] | training |
config | string | config.json | Path to JSON-formatted config file
print_model_summary | bool | False | Prints verbose summary of the model
load_checkpoint | string | None | Path to hdf5 file containing a snapshot of model weights

#### Additional arguments during inference:
Argument | Valid Inputs | Default | Description
-------- | ------------ | ------- | -----------
one_shot | bool | False | Denoises each audio file in a single forward propagation
target_field_length | int | as defined in config.json | Overrides parameter in config.json for denoising with different target-field lengths than used in training
batch_size | int | as defined in config.json | # of samples per batch
condition_value | int | 1 | Corresponds to speaker identity
clean_input_path | string | None | If supplied, SNRs of denoised samples are computed

Dataset
-----
The "Noisy speech database for training speech enhancement algorithms and TTS models" (NSDTSEA) is used for training the model. It is provided by the University of Edinburgh, School of Informatics, Centre for Speech Technology Research (CSTR).

1. [Download here](http://datashare.is.ed.ac.uk/handle/10283/1942)
2. Extract to `data/NSDTSEA`