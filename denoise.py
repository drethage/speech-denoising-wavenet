# A Wavenet For Speech Denoising - Dario Rethage - 19.05.2017
# Denoise.py

from __future__ import division
import os
import util
import tqdm
import numpy as np


def denoise_sample(model, input, condition_input, batch_size, output_filename_prefix, sample_rate, output_path):

    if len(input['noisy']) < model.receptive_field_length:
        raise ValueError('Input is not long enough to be used with this model.')

    num_output_samples = input['noisy'].shape[0] - (model.receptive_field_length - 1)
    num_fragments = int(np.ceil(num_output_samples / model.target_field_length))
    num_batches = int(np.ceil(num_fragments / batch_size))

    denoised_output = []
    noise_output = []
    num_pad_values = 0
    fragment_i = 0
    for batch_i in tqdm.tqdm(range(0, num_batches)):

        if batch_i == num_batches-1: #If its the last batch'
            batch_size = num_fragments - batch_i*batch_size

        condition_batch = np.array([condition_input, ] * batch_size, dtype='uint8')
        input_batch = np.zeros((batch_size, model.input_length))

        #Assemble batch
        for batch_fragment_i in range(0, batch_size):

            if fragment_i + model.target_field_length > num_output_samples:
                remainder = input['noisy'][fragment_i:]
                current_fragment = np.zeros((model.input_length,))
                current_fragment[:remainder.shape[0]] = remainder
                num_pad_values = model.input_length - remainder.shape[0]
            else:
                current_fragment = input['noisy'][fragment_i:fragment_i + model.input_length]

            input_batch[batch_fragment_i, :] = current_fragment
            fragment_i += model.target_field_length

        denoised_output_fragments = model.denoise_batch({'data_input': input_batch, 'condition_input': condition_batch})

        if type(denoised_output_fragments) is list:
            noise_output_fragment = denoised_output_fragments[1]
            denoised_output_fragment = denoised_output_fragments[0]

        denoised_output_fragment = denoised_output_fragment[:, model.target_padding: model.target_padding + model.target_field_length]
        denoised_output_fragment = denoised_output_fragment.flatten().tolist()

        if noise_output_fragment is not None:
            noise_output_fragment = noise_output_fragment[:, model.target_padding: model.target_padding + model.target_field_length]
            noise_output_fragment = noise_output_fragment.flatten().tolist()

        if type(denoised_output_fragments) is float:
            denoised_output_fragment = [denoised_output_fragment]
        if type(noise_output_fragment) is float:
            noise_output_fragment = [noise_output_fragment]

        denoised_output = denoised_output + denoised_output_fragment
        noise_output = noise_output + noise_output_fragment

    denoised_output = np.array(denoised_output)
    noise_output = np.array(noise_output)

    if num_pad_values != 0:
        denoised_output = denoised_output[:-num_pad_values]
        noise_output = noise_output[:-num_pad_values]

    valid_noisy_signal = input['noisy'][
                         model.half_receptive_field_length:model.half_receptive_field_length + len(denoised_output)]

    if input['clean'] is not None:
        input['noise'] = input['noisy'] - input['clean']

        valid_clean_signal = input['clean'][
                         model.half_receptive_field_length:model.half_receptive_field_length + len(denoised_output)]

        noise_in_denoised_output = denoised_output - valid_clean_signal

        rms_clean = util.rms(valid_clean_signal)
        rms_noise_out = util.rms(noise_in_denoised_output)
        rms_noise_in = util.rms(input['noise'])

        new_snr_db = int(np.round(util.snr_db(rms_clean, rms_noise_out)))
        initial_snr_db = int(np.round(util.snr_db(rms_clean, rms_noise_in)))

        output_clean_filename = output_filename_prefix + 'clean.wav'
        output_clean_filepath = os.path.join(output_path, output_clean_filename)
        util.write_wav(valid_clean_signal, output_clean_filepath, sample_rate)

        output_denoised_filename = output_filename_prefix + 'denoised_%ddB.wav' % new_snr_db
        output_noisy_filename = output_filename_prefix + 'noisy_%ddB.wav' % initial_snr_db
    else:
        output_denoised_filename = output_filename_prefix + 'denoised.wav'
        output_noisy_filename = output_filename_prefix + 'noisy.wav'

    output_noise_filename = output_filename_prefix + 'noise.wav'

    output_denoised_filepath = os.path.join(output_path, output_denoised_filename)
    output_noisy_filepath = os.path.join(output_path, output_noisy_filename)
    output_noise_filepath = os.path.join(output_path, output_noise_filename)

    util.write_wav(denoised_output, output_denoised_filepath, sample_rate)
    util.write_wav(valid_noisy_signal, output_noisy_filepath, sample_rate)
    util.write_wav(noise_output, output_noise_filepath, sample_rate)