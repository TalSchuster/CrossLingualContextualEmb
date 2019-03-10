import argparse
import numpy as np
import copy
import torch
from scipy.spatial.distance import cosine
from scipy.spatial import KDTree

from allennlp.commands.elmo import ElmoEmbedder

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--elmo_weights_path',
    type=str,
    default='models/$l_weights.hdf5',
    help="Path to elmo weights files - use $l as a placeholder for language.")
parser.add_argument(
    '--elmo_options_path',
    type=str,
    default='models/options262.json',
    help="Path to elmo options file. n_characters in the file should be 262")
parser.add_argument(
    '--align_path',
    type=str,
    default='models/align/$l_best_mapping.pth',
    help="Path to the aligning matrix saved in a pyTorch format. Use $l as a placeholder for language.")
parser.add_argument(
    '-l1',
    '--language1',
    type=str,
    default='en',
    help="language of sentence 1")
parser.add_argument(
    '-s1',
    '--sent1',
    type=str,
    default=
    'A house cat is valued by humans for companionship and for its ability to hunt rodents.',
    help="sentence in language 1")
parser.add_argument(
    '-w1',
    '--word1',
    type=str,
    default='cat',
    help=
    "Examined word from the sentence of language 1 (first occurrence will be used)"
)
parser.add_argument(
    '-l2',
    '--language2',
    type=str,
    default='es',
    help="language of sentence 2")
parser.add_argument(
    '-s2',
    '--sent2',
    type=str,
    default=
    'el gato doméstico está incluido en la lista 100 de las especies exóticas invasoras más dañinas del mundo.',
    help="sentence in language 2")
parser.add_argument(
    '-w2',
    '--word2',
    type=str,
    default='gato',
    help=
    "Examined word from the sentence of language 2 (first occurrence will be used)"
)
parser.add_argument(
    '--layer', type=int, default=1, help="Layer of Elmo to compute for")
parser.add_argument(
    '-c', '--cuda_device', type=int, default=-1, help="Cuda device")
args = parser.parse_args()


def parse_config(args):
    '''
    Replaces $l for the two languages.
    Prints the args
    '''

    new_args = copy.deepcopy(args)
    for k in vars(args):
        val = getattr(args, k)
        if type(val) is str and "$l" in val:
            new_val = val.replace("$l", args.language1)
            new_k = "{}_{}".format(k, "l1")
            setattr(new_args, new_k, new_val)

            new_val = val.replace("$l", args.language2)
            new_k = "{}_{}".format(k, "l2")
            setattr(new_args, new_k, new_val)

    print('-' * 30)
    for k in vars(new_args):
        print("{}: {}".format(k, getattr(new_args, k)))
    print('-' * 30)

    return new_args


def get_sent_embeds(sent, elmo_options_file, elmo_weights_file, layer,
                    cuda_device):
    '''
    Get the embeddings of the sentence words.
    sent - list of tokens
    elmo_options_file - json for model. n_characters should be 262
    elmo_weights_file - saved model
    layer - what layer of ELMo to output
    cuda_device - cuda device

    returns a numpy array with the embeddings per token for the selected layer
    '''
    elmo = ElmoEmbedder(elmo_options_file, elmo_weights_file, cuda_device)
    s_embeds = elmo.embed_sentence(sent)
    layer_embeds = s_embeds[layer,:,:]
    return layer_embeds

def analyze_sents(embeds_l1, embeds_l2, sent1, sent2, w1_ind, w2_ind, k=5):
    kdt = KDTree(embeds_l1)
    emb2 = embeds_l2[w2_ind]
    top_k_inds = kdt.query(emb2, k)[1]
    top_k_words = [sent1[i] for i in top_k_inds]
    print('Nearest {} neighbors for {} in "{}":\n{}'.format(k, sent2[w2_ind], ' '.join(sent1), ' ,'.join(top_k_words)))

    emb1 = embeds_l1[w1_ind]
    dist = cosine(emb1, emb2)
    print("Cosine distance between {} and {}: {}".format(sent1[w1_ind], sent2[w2_ind],dist))

if __name__ == '__main__':
    args = parse_config(args)

    # Language 1
    sent1_tokens = args.sent1.strip().split()
    w1_ind = sent1_tokens.index(args.word1)
    s1_embeds = get_sent_embeds(sent1_tokens, args.elmo_options_path,
                                args.elmo_weights_path_l1, args.layer,
                                args.cuda_device)

    align1 = torch.load(args.align_path_l1)
    s1_embeds_aligned = np.matmul(s1_embeds, align1.transpose())

    # Language 2
    sent2_tokens = args.sent2.strip().split()
    w2_ind = sent2_tokens.index(args.word2)
    s2_embeds = get_sent_embeds(sent2_tokens, args.elmo_options_path,
                                args.elmo_weights_path_l2, args.layer,
                                args.cuda_device)

    align2 = torch.load(args.align_path_l2)
    s2_embeds_aligned = np.matmul(s2_embeds, align2.transpose())

    # Analyse
    print("--- Before alignment:")
    analyze_sents(s1_embeds, s2_embeds, sent1_tokens, sent2_tokens, w1_ind, w2_ind)

    print("\n--- After alignment:")
    analyze_sents(s1_embeds_aligned, s2_embeds_aligned, sent1_tokens, sent2_tokens, w1_ind, w2_ind)
