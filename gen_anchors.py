import argparse
import numpy as np
import glob
import os
import json
import sys
from tqdm import tqdm

from allennlp.commands.elmo import ElmoEmbedder
from allennlp.common.util import lazy_groups_of
from allennlp.data import vocabulary

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--elmo_weights_path',
    type=str,
    default='models/$l_weights.hdf5',
    help="Path to elmo weight file. Can use $l as a placeholder for language argument")
parser.add_argument(
    '--elmo_options_path',
    type=str,
    default='models/options262.json',
    help="Path to elmo options file. n_characters in the file should be 262")
parser.add_argument(
    '-l',
    '--language',
    type=str,
    default='en',
    help="language to be used for paths")
parser.add_argument(
    '--txt_files',
    type=str,
    default='wiki_files/$l/dev*.txt',
    help=
    "Path to files with sentences (one per line). Can use $l as a placeholder for language argument"
)
parser.add_argument(
    '--vocab_file',
    type=str,
    default='vocabs/$l_50k.vocab',
    help=
    "Path to vocab file with tokens (one per line) to include in output. Should also include <UNK> token. Can use $l as a placeholder for language"
)
parser.add_argument(
    '--out_dir',
    type=str,
    default='anchors_output/$l',
    help="Path to output directory. Can use $l as a placeholder for language argument")
parser.add_argument(
    '--layers',
    type=int,
    nargs='+',
    default=[0, 1, 2],
    help="Layers of Elmo to store")
parser.add_argument(
    '-bs', '--batch_size', type=int, default=64, help="Batch size")
parser.add_argument(
    '-d', '--emb_dim', type=int, default=1024, help="Embeddings size")
parser.add_argument(
    '-c', '--cuda_device', type=int, default=-1, help="Cuda device. Use -1 for cpu")
args = parser.parse_args()


def parse_config(args):
    '''
    replace $l with args.language
    print args
    '''

    print('-' * 30)
    for k in vars(args):
        val = getattr(args, k)
        if type(val) is str and "$l" in val:
            val = val.replace("$l", args.language)

        setattr(args, k, val)
        print("{}: {}".format(k, getattr(args, k)))
    print('-' * 30)

    return args


def iter_line_words(f):
    '''
    Iterating over a text file line by line (each line is one sentence).
    Yielding sentence as list of words
    '''

    for line in f:
        yield line.strip().split()


def run_elmo(txt_files, elmo_options_file, elmo_weights_file, vocab, layers,
             batch_size, cuda_device):
    '''
    Running ELMo to compute anchors and norms per layer.
    txt_files - path to files with sentence per line (* in the path will be expended)
    elmo_options_file - json for model. n_characters should be 262
    elmo_weights_file - saved model
    vocab - file with token per word. Only those tokens will be saved
    layer - what layers to compute for (0 is uncontextualized layer)
    batch_size - batch size
    cuda_device - cuda device

    Returns dicts of anchors and norm (per layer) and the list of occurrences per token (indices by vocab)

    '''
    print('Loading ELMo Embedder...')
    elmo = ElmoEmbedder(elmo_options_file, elmo_weights_file, cuda_device)
    num_occurrences = [0] * vocab.get_vocab_size()
    anchors = {}
    norms = {}
    total_words = 0
    for l in layers:
        norms[l] = 0.0
        anchors[l] = np.zeros(
            shape=(vocab.get_vocab_size(), args.emb_dim))

    oov_ind = vocab.get_token_index(vocab._oov_token)
    shards = list(glob.glob(txt_files))
    for i, shard in enumerate(shards, start=1):
        print(
            ' --- Processing file %d out of %d: %s' % (i, len(shards), shard))
        num_lines = sum(1 for line in open(shard))
        f = open(shard, 'r', encoding='utf-8', newline='\n', errors='ignore')
        for batch in tqdm(
                lazy_groups_of(iter_line_words(f), batch_size),
                total=int(num_lines / batch_size)):
            embeds = elmo.embed_batch(batch)
            for sent, em in zip(batch, embeds):
                for j, w in enumerate(sent):
                    w_id = vocab.get_token_index(w)
                    if w_id == oov_ind:
                        continue

                    n = num_occurrences[w_id]
                    for l in layers:
                        anchors[l][
                            w_id, :] = anchors[l][w_id, :] * (
                                n / (n + 1)) + em[l, j, :] / (n + 1)
                        norm = np.linalg.norm(em[l,j,:])
                        norms[l] = norms[l] * (total_words / (total_words +
                                               1)) + norm / (total_words + 1)

                    total_words += 1
                    num_occurrences[w_id] += 1
        f.close()

    return anchors, norms, num_occurrences

def save_embeds(file_path, embeds, vocab, num_occurrences, emb_dim):
    # Don't include words not in the text.
    n_tokens = len(np.nonzero(num_occurrences)[0])
    with open(file_path, 'w') as f:
        f.write('%d %d\n' % (n_tokens, emb_dim))
        for i in range(embeds.shape[0]):
            if num_occurrences[i] == 0:
                continue

            token = vocab.get_token_from_index(i)
            to_dump = token + ' ' + ' '.join([str(v) for v in embeds[i, :]]) + '\n'
            f.write(to_dump)


if __name__ == '__main__':
    args = parse_config(args)
    if os.path.exists(args.out_dir):
        print("Output dir already exists: {}".format(args.out_dir))
        sys.exit(1)

    vocab = vocabulary.Vocabulary()
    vocab.set_from_file(args.vocab_file, oov_token='<UNK>')
    print("Loaded vocabulary of size {}".format(vocab.get_vocab_size()))

    anchors, norms, num_occurrences = run_elmo(
        args.txt_files, args.elmo_options_path, args.elmo_weights_path, vocab,
        args.layers, args.batch_size, args.cuda_device)

    os.makedirs(args.out_dir, exist_ok=True)
    norm_dict = {}
    print('Saving outputs to {}'.format(args.out_dir))
    for l in tqdm(args.layers):
        norm_key = 'avg_norm_layer_{}'.format(l)
        norm_dict[norm_key] = norms[l]
        file_path = os.path.join(args.out_dir, 'avg_embeds_{}.txt'.format(l))
        save_embeds(file_path, anchors[l], vocab, num_occurrences, args.emb_dim)

    file_path = os.path.join(args.out_dir, 'norms.json'.format(l))
    json.dump(norm_dict, open(file_path, 'w'))
