# CrossLingualELMo
Cross-Lingual Alignment of Contextual Word Embeddings

This repo will contain the code and models for the NAACL19 paper - [Cross-Lingual Alignment of Contextual Word Embeddings,  with Applications to Zero-shot Dependency Parsing](https://arxiv.org/abs/1902.09492)

More pieces of the code will be released soon.

## Updates:

* The multi-lingual parser code is now available at this [allennlp fork](https://github.com/TalSchuster/allennlp-MultiLang) (`requirements.txt` file of this repo is updated accordingly). See more details in the **Usage** section below.


# Aligned Multi Lingual Deep Contextual Word Embeddings

## Embeddings

The following are models were trained on Wikipedia and the second layer was aligned to English:

| Language        | Model weights | Aligning matrix  |
| ------------- |:-------------:| :-----:|
| en     | [weights.hdf5](https://www.dropbox.com/s/1h62kc1qdcuyy2u/en_weights.hdf5) | [best_mapping.pth](https://www.dropbox.com/s/nufj4pxxgv5838r/en_best_mapping.pth) |
| es     | [weights.hdf5](https://www.dropbox.com/s/ygfjm7zmufl5gu2/es_weights.hdf5) | [best_mapping.pth](https://www.dropbox.com/s/6kqot8ssy66d5u0/es_best_mapping.pth) |
| fr     | [weights.hdf5](https://www.dropbox.com/s/mm64goxb8wbawhj/fr_weights.hdf5) | [best_mapping.pth](https://www.dropbox.com/s/0zdlanjhajlgflm/fr_best_mapping.pth) |
| it     | [weights.hdf5](https://www.dropbox.com/s/owfou7coi04dyxf/it_weights.hdf5) | [best_mapping.pth](https://www.dropbox.com/s/gg985snnhajhm5i/it_best_mapping.pth) |
| pt     | [weights.hdf5](https://www.dropbox.com/s/ul82jsal1khfw5b/pt_weights.hdf5) | [best_mapping.pth](https://www.dropbox.com/s/skdfz6zfud24iup/pt_best_mapping.pth) |
| sv     | [weights.hdf5](https://www.dropbox.com/s/boptz21zrs4h3nw/sv_weights.hdf5) | [best_mapping.pth](https://www.dropbox.com/s/o7v64hciyifvs8k/sv_best_mapping.pth) |
| de     | [weights.hdf5](https://www.dropbox.com/s/2kbjnvb12htgqk8/de_weights.hdf5) | [best_mapping.pth](https://www.dropbox.com/s/u9cg19o81lpm0h0/de_best_mapping.pth) |


options file (for all models) - [options.json](https://www.dropbox.com/s/ypjuzlf7kj957g3/options262.json)

To download all the ELMo models in the table, use `get_models.sh`

To download all of the alignment matrices in the table, use `get_alignments.sh`.

### Generating anchors

Use the `gen_anchors.py` script to generate your own anchors. You will need a trained ELMo model, text files with one sentence per line, and vocab file with token per line containing the tokens that you wish to calculate for.
run `gen_anchors.py -h` for more details.

## Usage

### Generating aligned contextual embeddings

Given the output of a specific layer from ELMo (the contextual embeddings), run:
```
aligning  = torch.load(aligning_matrix_path)
aligned_embeddings = np.matmul(embeddings, aligning.transpose())
```

An example can be seen in `demo.py`. 

### Replicating the zero-shot cross-lingual dependency parsing results

1. Create an environment to install our fork of allennlp:

```
virtualenv -p /usr/bin/python3.6 allennlp_multilang
```
or, if you are using conda:
```
conda create -n allennlp_multilang python=3.6
```

2. Activate the environment and install:

```
source allennlp_multilang/bin/activate
pip install -r requirements.txt
```

3. Download the [uni-dep-tb](https://github.com/ryanmcd/uni-dep-tb) dataset (version 2) and follow the instructions to generate the [English PTB data](https://catalog.ldc.upenn.edu/LDC99T42)
4. Update the `allen_configs/multilang_dependency_parser.jsonnet` file with the path to dataset.
5. Train the model (the provided configuration is for 'es' as a target language):
```
allennlp train training_config/multilang_dependency_parser.jsonnet -s path_to_output_dir
```


### Using in any model

The aligments can be used with the [AllenNLP](https://allennlp.org) framework by simply using any model with ELMo embeddings and replacing the paths in the configuration with our provided models.

Each ELMo model was trained on Wikipedia of the relevant language. To align the models, you will need to add the following code to your model:

Load the alignment matrix in the `__init__()` function:

```
aligning_matrix_path = ... (pth file)
self.aligning_matrix = torch.FloatTensor(torch.load(aligning_matrix_path))
self.aligning = torch.nn.Linear(self.aligning_matrix[0], self.aligning_matrix[1], bias=False)
self.aligning.weight = torch.nn.Parameter(self.aligning_matrix, requires_grad=False)
```

Then, simply apply the alignment on the embedded tokens in the `forward()` pass:
```
embedded_text = self.aligning(embedded_text)
```

Note that our alignments were done on the second layer of ELMo but it can be learned and applied for each layer separately to preserve the original weighted sum of layers in the ELMo embedder. To fix the weighted sum across languages, use fixed parameters for `scalar_mix` (see our multi-lingual parser for an example).



# Citation

If you find this repo useful, please cite our paper.

```
@article{Schuster2019,
title = {Cross-Lingual Alignment of Contextual Word Embeddings, with Applications to Zero-shot Dependency Parsing},
author = {Schuster, Tal and Ram, Ori and Barzilay, Regina and Globerson, Amir},
eprint = {arXiv:1902.09492v1},
url = {https://arxiv.org/pdf/1902.09492.pdf},
year = {2019}
}
```
