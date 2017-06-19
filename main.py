# A Wavenet For Speech Denoising - Dario Rethage - 19.05.2017
# Main.py

import sys
import logging
import optparse
import json
import os
import models
import datasets
import util
import denoise


def set_system_settings():
    sys.setrecursionlimit(50000)
    logging.getLogger().setLevel(logging.INFO)


def get_command_line_arguments():
    parser = optparse.OptionParser()
    parser.set_defaults(config='config.json')
    parser.set_defaults(mode='training')
    parser.set_defaults(load_checkpoint=None)
    parser.set_defaults(condition_value=0)
    parser.set_defaults(batch_size=None)
    parser.set_defaults(one_shot=False)
    parser.set_defaults(clean_input_path=None)
    parser.set_defaults(noisy_input_path=None)
    parser.set_defaults(print_model_summary=False)
    parser.set_defaults(target_field_length=None)

    parser.add_option('--mode', dest='mode')
    parser.add_option('--print_model_summary', dest='print_model_summary')
    parser.add_option('--config', dest='config')
    parser.add_option('--load_checkpoint', dest='load_checkpoint')
    parser.add_option('--condition_value', dest='condition_value')
    parser.add_option('--batch_size', dest='batch_size')
    parser.add_option('--one_shot', dest='one_shot')
    parser.add_option('--noisy_input_path', dest='noisy_input_path')
    parser.add_option('--clean_input_path', dest='clean_input_path')
    parser.add_option('--target_field_length', dest='target_field_length')


    (options, args) = parser.parse_args()

    return options


def load_config(config_filepath):
    try:
        config_file = open(config_filepath, 'r')
    except IOError:
        logging.error('No readable config file at path: ' + config_filepath)
        exit()
    else:
        with config_file:
            return json.load(config_file)


def get_dataset(config, model):

    if config['dataset']['type'] == 'vctk+demand':
        return datasets.VCTKAndDEMANDDataset(config, model).load_dataset()
    elif config['dataset']['type'] == 'nsdtsea':
        return datasets.NSDTSEADataset(config, model).load_dataset()


def training(config, cla):

    # Instantiate Model
    model = models.DenoisingWavenet(config, load_checkpoint=cla.load_checkpoint, print_model_summary=cla.print_model_summary)
    dataset = get_dataset(config, model)

    num_train_samples = config['training']['num_train_samples']
    num_test_samples = config['training']['num_test_samples']
    train_set_generator = dataset.get_random_batch_generator('train')
    test_set_generator = dataset.get_random_batch_generator('test')

    model.fit_model(train_set_generator, num_train_samples, test_set_generator, num_test_samples,
                          config['training']['num_epochs'])


def get_valid_output_folder_path(outputs_folder_path):
    j = 1
    while True:
        output_folder_name = 'samples_%d' % j
        output_folder_path = os.path.join(outputs_folder_path, output_folder_name)
        if not os.path.isdir(output_folder_path):
            os.mkdir(output_folder_path)
            break
        j += 1
    return output_folder_path


def inference(config, cla):

    if cla.batch_size is not None:
        batch_size = int(cla.batch_size)
    else:
        batch_size = config['training']['batch_size']

    if cla.target_field_length is not None:
        cla.target_field_length = int(cla.target_field_length)

    if not bool(cla.one_shot):
        model = models.DenoisingWavenet(config, target_field_length=cla.target_field_length,
                                        load_checkpoint=cla.load_checkpoint, print_model_summary=cla.print_model_summary)
        print 'Performing inference..'
    else:
        print 'Performing one-shot inference..'

    samples_folder_path = os.path.join(config['training']['path'], 'samples')
    output_folder_path = get_valid_output_folder_path(samples_folder_path)

    #If input_path is a single wav file, then set filenames to single element with wav filename
    if cla.noisy_input_path.endswith('.wav'):
        filenames = [cla.noisy_input_path.rsplit('/', 1)[-1]]
        cla.noisy_input_path = cla.noisy_input_path.rsplit('/', 1)[0] + '/'
        if cla.clean_input_path is not None:
            cla.clean_input_path = cla.clean_input_path.rsplit('/', 1)[0] + '/'
    else:
        if not cla.noisy_input_path.endswith('/'):
            cla.noisy_input_path += '/'
        filenames = [filename for filename in os.listdir(cla.noisy_input_path) if filename.endswith('.wav')]

    clean_input = None
    for filename in filenames:
        noisy_input = util.load_wav(cla.noisy_input_path + filename, config['dataset']['sample_rate'])
        if cla.clean_input_path is not None:
            if not cla.clean_input_path.endswith('/'):
                cla.clean_input_path += '/'
            clean_input = util.load_wav(cla.clean_input_path + filename, config['dataset']['sample_rate'])

        input = {'noisy': noisy_input, 'clean': clean_input}

        output_filename_prefix = filename[0:-4] + '_'

        if config['model']['condition_encoding'] == 'one_hot':
            condition_input = util.one_hot_encode(int(cla.condition_value), 29)[0]
        else:
            condition_input = util.binary_encode(int(cla.condition_value), 29)[0]

        if bool(cla.one_shot):
            if len(input['noisy']) % 2 == 0:  # If input length is even, remove one sample
                input['noisy'] = input['noisy'][:-1]
                if input['clean'] is not None:
                    input['clean'] = input['clean'][:-1]
            model = models.DenoisingWavenet(config, load_checkpoint=cla.load_checkpoint, input_length=len(input['noisy']), print_model_summary=cla.print_model_summary)

        print "Denoising: " + filename
        denoise.denoise_sample(model, input, condition_input, batch_size, output_filename_prefix,
                                            config['dataset']['sample_rate'], output_folder_path)


def main():

    set_system_settings()
    cla = get_command_line_arguments()
    config = load_config(cla.config)

    if cla.mode == 'training':
        training(config, cla)
    elif cla.mode == 'inference':
        inference(config, cla)


if __name__ == "__main__":
    main()