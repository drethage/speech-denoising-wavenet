config.json - Configuring a training session
----

The parameters present in a `config.json` file allow one to configure a training session. Each of these parameters is described below:

### Dataset
How the data is used for training
* **extract_voice**: (bool) Only train on audio where voice is present
* **in_memory_percentage**: (float) Percentage of the dataset to load into memory, useful when dataset requires more memory than available
* **noise_only_percent**: (float) Percentage of training samples that should feature only noise at input and silence at output
* **num_condition_classes**: (int) Number of condition classes
* **path**: (string) Path to dataset
* **regain**: (float) RMS energy to which all samples should be regained to
* **sample_rate**: (int) Sample rate to which all samples should be resampled to
* **type**: (string) Identifier of which dataset is being used for training

### Model
What the model will be
* **condition_encoding**: (string) Which numerical representation to encode integer condition values to, either binary or one-hot
* **dilations**: (int) Maximum dilation factor as an exponent of 2, e.g. dilations = 9 results in a maximum dilation of 2^9 = 512
* **filters**:
  * **lengths**:
    * **res**: (int) Lengths of convolution kernels in residual blocks
    * **final**: ([int, int]) Lengths of convolution kernels in final layers, individually definable
    * **skip**: (int) Lengths of convolution kernels in skip connections
  * **depths**:
    * **res**: (int) Number of filters in residual-block convolution layers
    * **skip**: (int) Number of filters in skip connections
    * **final**: ([int, int]) Number of filters in final layers, individually definable
* **num_stacks**: (int) Number of stacks, as defined in the paper
* **target_field_length**: (int) Length of the output
* **target_padding**: (int) Number of samples used for padding the target_field *per side*

### Training
How training will be carried out

* **batch_size**: (int) Number of samples per batch
* **early_stopping_patience**: (int) Number of epochs to wait without improvement in accuracy before stopping training
* **loss**:
  * **out_1**: First term in the two term loss
    * **l1**: (float) Percentage weight given to L1 loss
    * **l2**: (float) Percentage weight given to L2 loss
    * **weight**: (float) Percentage weight given to first term
* **out_2**: Second term in two term loss
  * **l1**: (float) Percentage weight given to L1 loss
  * **l2**: (float) Percentage weight given to L2 loss
  * **weight**: (float) Percentage weight given to second term
* **num_epochs**: (int) Maximum number of epochs to train for
* **num_train_samples**: (int) Number of training samples presented to the model in one epoch,
* **num_test_samples**: (int) Number of test samples presented to the model in one epoch,
* **path**: (string) Path to the folder containing all files pertaining to the training session
* **verbosity**: (int) Keras verbosity level