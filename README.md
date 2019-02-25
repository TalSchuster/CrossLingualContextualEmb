# CrossLingualELMo
Cross-Lingual Alignment of Contextual Word Embeddings

This repo will contain the code and models for the NAACL19 paper - Cross-Lingual Alignment of Contextual Word Embeddings,  with Applications to Zero-shot Dependency Parsing (TODO -link)

More pieces of the code will be released soon. Meanwhile, the ELMo models for several languages with their alignment matrices are provided here. 


# Aligned Multi Lingual Deep Contextual Word Embeddings

## Embeddings

The following are models were trained on Wikipedia and the second layer was aligned to English:

| Language        | Model weights | Aligning matrix  |
| ------------- |:-------------:| :-----:|
| en     | [weights.hdf5](https://www.dropbox.com/s/1h62kc1qdcuyy2u/en_weights.hdf5?dl=0) | [best_mapping.pth](https://www.dropbox.com/s/nufj4pxxgv5838r/en_best_mapping.pth?dl=0) |
| es     | [weights.hdf5](https://www.dropbox.com/s/ygfjm7zmufl5gu2/es_weights.hdf5?dl=0) | [best_mapping.pth](https://www.dropbox.com/s/6kqot8ssy66d5u0/es_best_mapping.pth?dl=0) |
| fr     | [weights.hdf5]()(add) | [best_mapping.pth]()(add) |
| it     | [weights.hdf5]()(add) | [best_mapping.pth]()(add) |
| pt     | [weights.hdf5](https://www.dropbox.com/s/ul82jsal1khfw5b/pt_weights.hdf5?dl=0) | [best_mapping.pth](https://www.dropbox.com/s/skdfz6zfud24iup/pt_best_mapping.pth?dl=0) |
| sv     | [weights.hdf5]()(add) | [best_mapping.pth]()(add) |


options file (for all models) - [options.json](https://www.dropbox.com/s/ypjuzlf7kj957g3/options262.json?dl=0)

## Usage

The models can be used with the [AllenNLP](https://allennlp.org) framework by simply using any model with ELMo embeddings and replacing the paths in the configuration with our provided models.

Each ELMo model was trained on Wikipedia of the relevant language. To align the models, you will need to add the following code to your model:

Load the alignment matrix in the `__init__()` function:

```
aligning_matrix_path = ...
self.aligning_matrix = torch.FloatTensor(torch.load(aligning_matrix_path))
self.aligning = torch.nn.Linear(self.aligning_matrix[0], self.aligning_matrix[1], bias=False)
self.aligning.weight = torch.nn.Parameter(self.aligning_matrix)
self.aligning.weight.requires_grad = False
```

Then, simply apply the alignment on the embedded tokens:
```
embedded_text = self.aligning(embedded_text)
```

Note that our alignments were done on the second layer of ELMo and we had to do a few hacks to freeze the layer weights in the AllenNLP repo. We will release that code soon. However, the alignment can be done for each layer separately to preserve the original weighted sum of layers in the ELMo embedder.



# Citation

TODO
