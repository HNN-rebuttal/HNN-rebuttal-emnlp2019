import argparse
from io_utils import *
from scorer import scorer
from bilingual_space import bilingual_space
try:
    import cupy as xp
    gpu = True
    import numpy as np
except ImportError:
    import numpy as xp
    gpu = False
    np = xp

def parse_args():
    parser = argparse.ArgumentParser(
                formatter_class=argparse.RawTextHelpFormatter)
    # data loading
    data_args = parser.add_argument_group('Data Options')
    data_args.add_argument('src_emb', type=str,
                           help='path to src embedding file')
    data_args.add_argument('tgt_emb', type=str,
                           help='path to tgt embedding file')
    data_args.add_argument('--src_vocab_file', type=str,
                           help='If given, src vocab is a intersection of '
                           'words that appear both in the src_emb_file and '
                           'src_vocab_file')
    data_args.add_argument('--tgt_vocab_file', type=str,
                           help='If given, tgt vocab is a intersectoin of '
                           'words that appear both in the src_emb_file and '
                           'tgt_vocab_file.')
    data_args.add_argument('-m', '--src_vocab_size', type=int, default=None,
                           help='source vocab size')
    data_args.add_argument('-n', '--tgt_vocab_size', type=int, default=None,
                           help='target vocab size')
    data_args.add_argument('-d', '--seed_dict', type=str,
                           help='path to training dictionary file')
    data_args.add_argument('-t', '--test_dict', type=str, nargs='+',
                           help='path to test dictionary file, the overlap '
                           'with training dict will be\n'
                           'removed when calculating translation accuracy. \n'
                           'Allows multiple test dictionary, and report '
                           'accuracy on each')
    data_args.add_argument('--max_query', type=int,
                           help='maximum number of queries if given')

    # Linear Mapping Options
    mapping_args = parser.add_argument_group('Linear Mapping Options')
    mapping_args.add_argument('--loss_type', type=str,
                              choices=['l2', 'hinge'], default='l2',
                              help='Choice of Linear Mapping Learner:\n'
                              'l2: min \sum_i |W * x_i - y_i|^2 \n'
                              'hinge: min \sum_{i,j} '
                              'max{0, th + d(W * x_i, y_j) - d(W * x_i, y_i)}')
    mapping_args.add_argument('--orth', action='store_true',
                              help='whether to contrain the mapping to be '
                              'orthonormal.')
    mapping_args.add_argument('--hinge_epoch', type=int, default=10,
                              help='number of epochs in SGD for hinge loss')
    mapping_args.add_argument('--hinge_dist', choices=['sqeuc', 'cos'],
                              default='cos',
                              help='The distance `d(a, b)` in hinge loss')
    mapping_args.add_argument('--hinge_lr', type=float, default=1e-5,
                              help='learning rate if using hinge loss to '
                              'learn the mapping')
    mapping_args.add_argument('--hinge_batch', type=int, default=64,
                              help='batch size when minimizing hinge loss')
    mapping_args.add_argument('--hinge_sample', choices=['random', 'top'],
                              default='top',
                              help='Choice of negative sampling in minimizing '
                              'hinge loss.')
    mapping_args.add_argument('--hinge_ns', type=int, default=10,
                              help='number of negative samples for hinge loss')
    mapping_args.add_argument('--hinge_alpha', type=float, default=0.5,
                              help='for computing threshold in hinge loss. '
                              'Refer to bilingual_space class')
    mapping_args.add_argument('--hinge_th', type=float,
                              help='constant threshold in hinge loss')

    # Retrieval Options
    retrieval_args = parser.add_argument_group('Retrieval Options')
    retrieval_args.add_argument('--metric', type=str,
                                choices=['cos', 'sqeuc'], default='cos',
                                help='choice of distance metric for retrieval.'
                                ' Default to cosine distance')
    retrieval_args.add_argument('--method', type=str,
                                choices=['nn', 'csls', 'isf', 'hnn'],
                                default='nn',
                                help='Choice of retrieval method.\n'
                                'nn: Vanilla nearest neighbor\n'
                                'csls: Cross-domain Local Scaling\n'
                                'isf: inverted softmax\n'
                                'hnn: Hubless Nearest Neighbor\n'
                                'Default to nn.')
    retrieval_args.add_argument('--batch', type=int, default=128,
                                help='computing distances in batch of '
                                'queries or gallery examples, to avoid '
                                'memory overflow')
    retrieval_args.add_argument('--knn', type=int, default=10,
                                help='number of nearest neighbors to estimate '
                                'hubness, parameter for csls only. '
                                'Default to 10')
    retrieval_args.add_argument('--epsilon', type=float, default=1./30,
                                help='heat kernel parameter for '
                                'inverted softmax and HNN. Default to 1/30')
    retrieval_args.add_argument('--iters', type=int, default=30,
                                help='number of batch gradient steps in '
                                'HNN solver. Default: 30')
    retrieval_args.add_argument('--lr', type=float, default=1e4,
                                help='learning rate for gradient steps in HNN')
    
    # Logging and Checkpoint Options
    ckpt_args = parser.add_argument_group('Logging and Checkpoint Options')
    ckpt_args.add_argument('--task_name', type=str, help='name the task')
    ckpt_args.add_argument('--hinge_training_loss', type=str,
                           help='path to save training loss if hinge loss')
    ckpt_args.add_argument('--save_translation', type=str,
                           help='path to save top-10 '
                           'translations for source words')
    ckpt_args.add_argument('--only_test_translation', action='store_true',
                           help='If set, save translations only for '
                           'words that are in the test dictionary.\n'
                           'Multiple test dictionaries will be merged into '
                           'a single one.\n'
                           'Otherwise, save all translations for '
                           'source words in source vocab.\n'
                           'This may take significantly longer time.')

    args = parser.parse_args()

    # print setups
    _, f = os.path.split(args.seed_dict)
    task = args.task_name if args.task_name else f.split('.')[0]
    print("Task: {}".format(task), flush=True)
    print("GPU: {}".format(gpu), flush=True)
    print("Src_emb_file: {}".format(args.src_emb), flush=True)
    print("Tgt_emb_file: {}".format(args.tgt_emb), flush=True)
    print("Src_vocab_file: {}".format(args.src_vocab_file), flush=True)
    print("Tgt_vocab_file: {}".format(args.tgt_vocab_file), flush=True)
    print("Max_src_vocab_size: {}".format(args.src_vocab_size), flush=True)
    print("Max_tgt_vocab_size: {}".format(args.tgt_vocab_size), flush=True)
    print("Seeding dictionary: {}".format(args.seed_dict), flush=True)
    print("Ground-truth test dictionaries:\n{}".format(
          '\n'.join(args.test_dict)), flush=True)
    print("Upper limit on the number of query items: {}".format(
          args.max_query), flush=True)

    print("loss type: {}".format(args.loss_type), flush=True)
    print("orthogonal constraint: {}".format(args.orth), flush=True)
    if args.loss_type == 'hinge':
        print("Hinge loss hyper-params:", flush=True)
        print("  distance: {}".format(args.hinge_dist), flush=True)
        if args.hinge_th is not None:
            print("  threshold: {}".format(args.hinge_th), flush=True)
        else:
            print("  alpha quantile: {}".format(args.hinge_alpha), flush=True)
        print("  #negative samples: {}".format(args.hinge_ns), flush=True)
        print("  sample method: {}".format(args.hinge_sample), flush=True)
        print("  learning rate: {}".format(args.hinge_lr), flush=True)
        print("  batch size: {}".format(args.hinge_batch), flush=True)
        print("  epoch: {}".format(args.hinge_epoch), flush=True)

    print("Retrieval metric: {}".format(args.metric), flush=True)
    print("Retrieval method: {}".format(args.method), flush=True)
    if args.method == 'isf':
        print("Entropy regularizer: {}".format(args.epsilon), flush=True)
    elif args.method == 'hnn':
        print("Entropy regularizer: {}".format(args.epsilon), flush=True)
        print("Learning rate: {}".format(args.lr), flush=True)
        print("Number of iterations {}".format(args.iters), flush=True)

    if args.loss_type=='hinge' and args.hinge_training_loss:
        print("Save hinge training loss to {}".format(
              args.hinge_training_loss), flush=True)
    if args.save_translation is not None:
        if args.only_test_translation:
            print("Save top-10 translations to {} for all test words".format(
                  args.save_translation), flush=True)
        else:
            print("Save top-10 translations to {} for all source words".format(
                  args.save_translation), flush=True)
    print()
    return args

def main():
    args = parse_args()

    print("Loading source embeddings and building source vocab ...",
          flush=True)
    src_emb, src_vocab = load_embedding(args.src_emb, args.src_vocab_size,
                                        args.src_vocab_file)
    print("Loaded source vocab size: {}".format(len(src_vocab)), flush=True)

    print("Loading target embeddings and building target vocab ...",
          flush=True)
    tgt_emb, tgt_vocab = load_embedding(args.tgt_emb, args.tgt_vocab_size,
                                        args.tgt_vocab_file)
    print("Loaded target vocab size: {}".format(len(tgt_vocab)), flush=True)

    space = bilingual_space(src_emb, tgt_emb)

    print("Loading seeding dictionary ...", flush=True)
    train_src_words, train_tgt_words = load_seeding_dict(args.seed_dict,
                                                         src_vocab, tgt_vocab) 

    print("Learning Linear Mapping ...", flush=True)
    W = space.learn_mapping([train_src_words, train_tgt_words],
                            args.loss_type,
                            orth=args.orth,
                            epochs=args.hinge_epoch,
                            dist=args.hinge_dist,
                            lr=args.hinge_lr,
                            seed_per_batch=args.hinge_batch,
                            sample_method=args.hinge_sample,
                            samples=args.hinge_ns,
                            alpha=args.hinge_alpha,
                            constant_th=args.hinge_th)

    if args.hinge_training_loss is not None:
        np.save(args.hinge_training_loss, np.array(space.loss))

    print("Building {} scorer ...".format(args.method.upper()), flush=True)
    S = scorer(src_emb.dot(W), tgt_emb, src_vocab, tgt_vocab)
    S.build_translator(args.metric, args.method,
                       k=args.knn, epsilon=args.epsilon, batch=args.batch,
                       iters=args.iters, lr=args.lr)

    print("Querying ...", flush=True)
    for test_dict_file in args.test_dict:
        print("test_dict_file: {}".format(test_dict_file), flush=True)
        queries = load_queries(test_dict_file, src_vocab, tgt_vocab,
                               train_src_words, args.max_query)
        test_precision = S.translate(queries, args.save_translation,
                                     args.only_test_translation)
        print("----Retrieval accuracy ----", flush=True)
        print("P@1(%): {}, P@5(%): {}, P@10(%): {}".format(
              test_precision[0], test_precision[1], test_precision[2]),
          flush=True)
    print("Done", flush=True) 

if __name__ == '__main__':
    main()
