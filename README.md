# CrossLingualELMo
Cross-Lingual Alignment of Contextual Word Embeddings

This repo will contain the code and models for the NAACL19 paper - [Cross-Lingual Alignment of Contextual Word Embeddings,  with Applications to Zero-shot Dependency Parsing](https://arxiv.org/abs/1902.09492)

More pieces of the code will be released soon.

## Updates:

* Computed anchors for English (to help with the alignment computation for more languages)

* Alignment matrices for all layers of ELMO.

* A script to compute anchors for a BERT model is now available.

* The Multilingual ELMo is now merged to the [AllenNLP framework](https://github.com/allenai/allennlp) (version >= 0.8.5). Anchors for other models can be computed using the code here.

* <del> We are working on merging the Multilingual ELMo to the AllenNLP framework. Hopefully we will get to finish it soon.

* <del> The multi-lingual parser code is now available at this [allennlp fork](https://github.com/TalSchuster/allennlp-MultiLang) (`requirements.txt` file of this repo is updated accordingly). See more details in the **Usage** section below.


# Aligned Multi Lingual Deep Contextual Word Embeddings

## Embeddings

The following models were trained on Wikipedia. We provide the alignment of the first LSTM output of ELMo to English. The English file contains the identity matrix divided by the average norm for that layer.

| Language        | Model weights | Alignment matrix (First LSTM layer) *  |
| ------------- |:-------------:| :-----:|
| English     | [weights.hdf5](https://www.dropbox.com/s/1h62kc1qdcuyy2u/en_weights.hdf5) | [en_best_mapping.pth](https://www.dropbox.com/s/nufj4pxxgv5838r/en_best_mapping.pth) |
| Spanish     | [weights.hdf5](https://www.dropbox.com/s/ygfjm7zmufl5gu2/es_weights.hdf5) | [es_best_mapping.pth](https://www.dropbox.com/s/6kqot8ssy66d5u0/es_best_mapping.pth) |
| French     | [weights.hdf5](https://www.dropbox.com/s/mm64goxb8wbawhj/fr_weights.hdf5) | [fr_best_mapping.pth](https://www.dropbox.com/s/0zdlanjhajlgflm/fr_best_mapping.pth) |
| Italian     | [weights.hdf5](https://www.dropbox.com/s/owfou7coi04dyxf/it_weights.hdf5) | [it_best_mapping.pth](https://www.dropbox.com/s/gg985snnhajhm5i/it_best_mapping.pth) |
| Portuguese     | [weights.hdf5](https://www.dropbox.com/s/ul82jsal1khfw5b/pt_weights.hdf5) | [pt_best_mapping.pth](https://www.dropbox.com/s/skdfz6zfud24iup/pt_best_mapping.pth) |
| Swedish     | [weights.hdf5](https://www.dropbox.com/s/boptz21zrs4h3nw/sv_weights.hdf5) | [sv_best_mapping.pth](https://www.dropbox.com/s/o7v64hciyifvs8k/sv_best_mapping.pth) |
| German     | [weights.hdf5](https://www.dropbox.com/s/2kbjnvb12htgqk8/de_weights.hdf5) | [de_best_mapping.pth](https://www.dropbox.com/s/u9cg19o81lpm0h0/de_best_mapping.pth) |

\* Alignments for layer 0 (pre LSTM) and layer 2 (post LSTM) for all above languages - [alignments_0_2.zip](https://www.dropbox.com/s/ymnyptj3lupvcw7/alignments_0_2.zip)

* Unsupervised alignments for layer 1 - [alignments_unsupervised.zip](https://www.dropbox.com/s/sgi86uc8stu70bg/alignments_unsupervised.zip)

* Options file (for all models) - [options.json](https://www.dropbox.com/s/ypjuzlf7kj957g3/options262.json)

* Computed anchors for the Enlgish model - [english_anchors.zip](https://www.dropbox.com/s/8ad5oqhbh3xlnnf/english_anchors.zip)

#### Download helpers:

* To download all the ELMo models in the table, use `get_models.sh`

* To download all of the alignment matrices in the table, use `get_alignments.sh`.

* Alternatively, If you are interested in applying it in an Allennlp model, you can just add the path to the configuration file (check the examples in `allen_configs`)
### Generating anchors

In order to generate your own anchors - use the `gen_anchors.py` script to generate your own anchors. You will need a trained ELMo model, text files with one sentence per line, and vocab file with token per line containing the tokens that you wish to calculate for.
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
virtualenv -p /usr/bin/python3.6 allennlp_env
```
or, if you are using conda:
```
conda create -n allennlp_env python=3.6
```

2. Activate the environment and install allennlp:

```
source allennlp_env/bin/activate
pip install -r requirements.txt
```

3. Download the [uni-dep-tb](https://github.com/ryanmcd/uni-dep-tb) dataset (version 2) and follow the instructions to generate the [English PTB data](https://catalog.ldc.upenn.edu/LDC99T42)
4. Train the model (the provided configuration is for 'es' as a target language):
```
TRAIN_PATHNAME='universal_treebanks_v2.0/std/**/*train.conll' \
DEV_PATHNAME='universal_treebanks_v2.0/std/**/*dev.conll' \
TEST_PATHNAME='universal_treebanks_v2.0/std/**/*test.conll' \
allennlp train training_config/multilang_dependency_parser.jsonnet -s path_to_output_dir;
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




# Citation

If you find this repo useful, please cite our paper.

```
@InProceedings{Schuster2019,
    title = "Cross-Lingual Alignment of Contextual Word Embeddings, with Applications to Zero-shot Dependency Parsing",
    author = "Schuster, Tal  and
      Ram, Ori  and
      Barzilay, Regina  and
      Globerson, Amir",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N19-1162",
    pages = "1599--1613"
}
```
