try:
    import cupy as xp
    gpu = True
except ImportError:
    import numpy as xp
    gpu = False
from random import shuffle, seed
from function_utils import distance_function, top_k 
from scorer import scorer 

seed(777)
eps=xp.finfo(float).eps

class bilingual_space:
    def __init__(self, src_space, tgt_space):
        self.src_space = src_space
        self.tgt_space = tgt_space
        self.src_size = src_space.shape[0]
        self.tgt_size = tgt_space.shape[0]

    def learn_mapping(self, seeds, loss_type, **kwargs):
        """
            Learn the mapping function. The objective is l2 loss or
            hinge loss. Orthogonal constraint is optional.
            
            -Input:
            seeds: A list of two lists. seeds[0][i] and seeds[1][i] specifies
                   a seeding pair

            loss_type: 'l2' or 'hinge'
              'l2': min_R \sum_i |R * x_i - y_i|^2
              'hinge': min_R \sum_i \sum_{j!=i}
                          max{0, th_i + |R * x_i - y_i|^2 - |R * x_i - y_j|^2}
                       The objetive is optimzied by SGD. Each iteration samples
                       some random negative examples

            kwargs: misc parameters, mostly for hinge loss minimizer
              orth: (False) whether to contrain the mapping being orthogonal
              epochs: number of epochs in SGD optimizer
                     if loss_type='hinge'
              seed_per_batch, number of seeds per minibatch in SGD
              lr: learning rate
              dist: the distance (cosine or squared Euclidean) in hinge loss.
                    Cosine is suggested.
              samples: number of negative samples per seeding pair
              alpha: determine threshold `th_i` in the following way
                     th_i = percentile(
                            d(R * x_i, y_j) - d(R * x_i, y_i), alpha)
              constant_th: constant threshold for all pairs. 
                     If given, alpha is futile. 
              src_vocab, tgt_vocab: source and target vocabs. If given, will
                    report P@1 during the minimization of hinge loss
              queries: for reporting test accuracy, src_vocab and tgt_vocab 
                    must be given
            -Output:
            W: linear mapping
        """
        # prepare default params
        orth = kwargs.get('orth')
        epochs = kwargs.get('epochs')
        dist = kwargs.get('dist')
        lr = kwargs.get('lr')
        s_per_b = kwargs.get('seed_per_batch')
        sample_method = kwargs.get('sample_method')
        ns = kwargs.get('samples')
        alpha = kwargs.get('alpha')
        constant_th = kwargs.get('constant_th')
        src_vocab = kwargs.get('src_vocab', None)
        tgt_vocab = kwargs.get('tgt_vocab', None)
        queries = kwargs.get('queries', None)
        seed_dict = {}
        for i, j in zip(*seeds):
            if i not in seed_dict:
                seed_dict[i] = [j]
            else:
                seed_dict[i].append(j)

        if orth:
            C = self.src_space[seeds[0]].T.dot(self.tgt_space[seeds[1]])
            U, _, Vh = xp.linalg.svd(C)
            self.W = U.dot(Vh)
        else:
            if not gpu:
                self.W = xp.linalg.lstsq(self.src_space[seeds[0]],
                                    self.tgt_space[seeds[1]],
                                    rcond=None)[0]
            else:
                self.W = xp.linalg.pinv(self.src_space[seeds[0]]).dot(
                            self.tgt_space[seeds[1]])
        
        if loss_type == 'l2':
            self.loss = xp.linalg.norm(
                        self.src_space[seeds[0]].dot(self.W)\
                        - self.tgt_space[seeds[1]]) ** 2
        elif loss_type == 'hinge':
            # SGD optimizer, each iteration samples seeds and negative samples
            # Initialized with l2 solution
            self.loss = []
            dim = self.src_space.shape[1]
            total_it = 0
            th = self.determine_th(seeds, alpha, dist, constant_th)
            for ep in range(epochs):
                S = [_ for _ in zip(*(seeds + [th]))]
                shuffle(S)
                for it in range(0, len(S), s_per_b):
                    i_s, i_t, th_i = zip(*S[it:min(len(S), it+s_per_b)])
                    i_s, i_t, th_i = list(i_s), list(i_t), xp.array(th_i)
                    B = len(i_s)
                    Wx = self.src_space[i_s].dot(self.W)
                    D = distance_function(Wx, self.tgt_space, dist)
                    if sample_method == 'random':
                        j = [xp.random.choice(
                             [_ for _ in range(self.tgt_size)\
                                if _ not in seed_dict[i_s_]],
                                ns).tolist() for i_s_ in i_s]
                    elif sample_method == 'top':
                        j = []
                        for ii, i_s_ in enumerate(i_s):
                            d_ii = xp.copy(D[ii])
                            d_ii[seed_dict[i_s_]] = xp.float('inf')
                            j.append(
                                top_k(d_ii, ns, biggest=False)[0].tolist())

                    delta = D[xp.tile(range(B), (ns, 1)).T, j]\
                            - D[xp.arange(B), i_t][:, None]

                    # print some diagnostics
                    ell = xp.sum(xp.maximum(th_i[:, None] - delta, 0))/B
                    if gpu:
                        ell = ell.get()
                    self.loss.append(ell)
                    if total_it % 100 == 0:
                        if all([src_vocab, tgt_vocab, queries]):
                            P_at_1 = self.report_precision(
                                        src_vocab, tgt_vocab, queries)
                            p_str = ', p@1 {}%'.format(P_at_1[0])
                        else:
                            p_str = ''
                        print("Epoch {}, Iter {}, loss {:.2f}".format(
                              ep, total_it, ell) + p_str, flush=True)

                    incur_loss = delta < th_i[:, None]
                    n_incur = [xp.sum(xp.array(_)) for _ in incur_loss]
                    if dist == 'sqeuc':
                        delta_y = [xp.sum(self.tgt_space[j[_]][incur_loss[_]],\
                                axis=0) - self.tgt_space[i_t[_]] * n_incur[_]\
                                for _ in range(B)]
                        grad = self.src_space[i_s].T.dot(xp.vstack(delta_y))
                    elif dist == 'cos':
                        Wx_norm = xp.linalg.norm(Wx, ord=2, axis=1, keepdims=1)
                        delta_y = [
                            xp.sum(self.tgt_space[j[_]][incur_loss[_]] / (eps+\
                            xp.linalg.norm(self.tgt_space[j[_]][incur_loss[_]],
                            ord=2, axis=1, keepdims=1)), axis=0) -\
                            self.tgt_space[i_t[_]] /\
                            xp.linalg.norm(self.tgt_space[i_t[_]]) * n_incur[_]
                            for _ in range(B)
                            ]
                        delta_cos = [xp.sum(delta[_][incur_loss[_]])\
                                     for _ in range(B)] 
                        grad = (self.src_space[i_s] / Wx_norm).T.dot(
                                xp.vstack(delta_y)) +\
                                (self.src_space[i_s] * xp.vstack(delta_cos)).T\
                                .dot(Wx/Wx_norm)
                    if orth:
                        # Use Cayley transform to maintain orthogonality
                        A = grad.dot(self.W.T)
                        A = A - A.T
                        Q = xp.linalg.inv(xp.eye(dim) + lr/2 * A).dot(
                            xp.eye(dim) - lr/2 * A)
                        self.W = Q.dot(self.W)
                    else:
                        self.W -= lr * grad / B
                    
                    total_it += 1
        return self.W

    def determine_th(self, seeds, alpha, dist, constant_th):
        S = len(seeds[0])
        if constant_th is not None:
            return (xp.ones(S) * constant_th).tolist()
        th = []
        for i in range(0, S, 128):
            sl = slice(i, min(i+128, S))
            D = distance_function(self.src_space[seeds[0][sl]].dot(self.W),
                                  self.tgt_space, dist)
            th_i = xp.percentile(
                D - D[xp.arange(D.shape[0]), seeds[1][sl]][:, None],
                alpha*100, axis=1)
            th.append(th_i)
        return xp.hstack(th).tolist()

    def report_precision(self, src_vocab, tgt_vocab, queries):
        S = scorer(self.src_space.dot(self.W), self.tgt_space,
                   src_vocab, tgt_vocab)
        S.build_translator('cos', 'nn')
        return S.translate(queries)
