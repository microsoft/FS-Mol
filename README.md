# FS-Mol: A Few-Shot Learning Dataset of Molecules

This repository contains code and data for FS-Mol: A Few-Shot Learning Dataset of Molecules. 

# Installation

1. Clone or download this repository
2. Install dependencies
   ```
   cd FS-Mol

   conda env create -f environment.yml
   conda activate fsmol

   ```

## FSMolDataset
The `FSMolDataset` provides access to the train/valid/test tasks of the few-shot dataset. An instance is created from the data directory by `FSMolDataset.from_directory(/path/to/dataset)` and access to the iterable over task files is given by `FSMolDataset.get_task_reading_iterable()`. This allows specification of a callable to allow trasnformations while reading a list of task files, and permits multithreaded data loading. The default implementation returns an iterable over `FSMolTask` objects, which each contain an entire single task's set of molecules and labels. These are held by the `MoleculeDatapoint` objects. More details and examples of how to use `FSMolDataset` are available in `fs_mol/notebooks/dataset.ipynb`.

## Available Model Implementations

We provide implementations for three key few-shot learning methods: Multitask learning, Model-Agnostic Meta-Learning, and Prototypical Networks, as well as evaluation on the Single-Task baselines and the Molecule Attention Transformer (MAT) [paper](https://arxiv.org/abs/2002.08264v1), [code](https://github.com/lucidrains/molecule-attention-transformer). 

These baseline methods can be run on the FS-Mol dataset as follows:

### kNNs and Random Forests -- Single Task Baselines

The baseline single-task evaluation can be run as follows, with a choice of kNN or randomForest model:

```bash
python fs_mol/baseline_test.py /path/to/data --model {kNN, randomForest}
```

### MAT -- Single Task Methods

The Molecule Attention Transformer can be evaluated as:

```bash
python fs_mol/mat_test.py /path/to/pretrained-mat /path/to/data
```

### GNN-MAML pre-training and evaluation

The current defaults were used to train the final versions of GNN-MAML available here. 

```bash
python fs_mol/maml_train.py /path/to/data 
```

Evaluation is run as: 

```bash
python fs_mol/maml_test.py /path/to/data --trained_model /path/to/gnn-maml-checkpoint
```

### GNN-MT pre-training and evaluation

```bash
python fs_mol/multitask_train.py /path/to/data 
```

Evaluation is run as: 

```bash
python fs_mol/multitask_test.py /path/to/gnn-mt-checkpoint /path/to/data
```

### Prototypical Networks (PN) pre-training and evaluation

```bash
python fs_mol/protonet_train.py /path/to/data 
```

Evaluation is run as: 

```bash
python fs_mol/protonet_test.py /path/to/pn-checkpoint /path/to/data
```

## Available Model Checkpoints

We provide pre-trained models for `GNN-MAML`, `GNN-MT` and `PN`.

| Model Name | Description | Checkpoint File |
|------------|-------------|-----------------|
| GNN-MAML   |             |                 |
|------------|-------------|-----------------|
| GNN-MT     |             |                 |
|------------|-------------|-----------------|
| PN         |             |                 |
|------------|-------------|-----------------|



## Specifying, Training and Evaluating New Model Implementations

Flexible definition of few-shot models and single task models is defined as demonstrated in the range of train and test scripts in `fs_mol`. 

We give a detailed example of how to use the abstract class `AbstractTorchFSMolModel` in `notebooks/integrating_torch_models.ipynb` to integrate a new general PyTorch model, and note that the evaluation procedure described below is demonstrated on `sklearn` models in `fs_mol/baseline_test.py` and on a Tensorflow-based GNN model in `fs_mol/maml_test.py`.

### Running the benchmark

We note that the benchmark evaluation method, in particular the use of `eval_model` in conjunction with the `FSMolDataset`, is designed to be general and applicable to a wide range of models regardless of preferred implementation library. `notebooks/evaluation.ipynb` describes how to integrate the benchmarking procedure and run for direct comparison on the FS-Mol benchmark tasks.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
