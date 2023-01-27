"""
corpus.py

Data Structure for managing corpus of PCFG. Stores all possible
sentences mapped to their meanings as syntactic trees.
Used by generator.py to generate sentences with 8 complexity metrics:

Surprisal (Lexical)
Lexical Entropy
Structural Entropy (over meanings)
Lexical Entropy Reduction
Structural Entropy Reduction
Kulback-Liebler Divergence (Pk || Pk-1) over meanings
Kulback-Liebler Divergence (Pk-1 || Pk) over meanings
Mutual Information
"""

import numpy as np
import itertools
import pickle
import pandas as pd


class SyntaxTree:
    # A DS for representing meaning
    def __init__(self, log_likelihood):
        """
        Currently, all that is needed to calculate complexity metrics of
        sentences in the Corpus is the likelihood of the sentence occuring
        given the grammar. Because of that I'm going to leave representing
        meaning for later; this DS just stores the probability of the sentence.

        :param likelihood: float
        """

        self.log_likelihood = log_likelihood

# NOTE: consider making SyntaxForest, a container object for SyntaxTrees,
#       that stores the (log) total probability of all the meanings it
#       contains.

class Corpus:
    # Corpus DS
    def __init__(self, nonterminals, sentences, meanings, name):
        """
        R: sentences and meanings are coindexed.
           nonterminals contains all strings present in sentences, no more and
           no less.
        E: Constructs a (sorted) Corpus using the sentences and meanings.

        :param nonterminals: iterable of strings
        :param sentences: ordered collection of lists of strings
        :param meanings: List of SyntaxTrees
        :param name: string name of corpus, used for saving to pickle binary
        """

        # store name to identify corpus for pickle serialization
        self.name = name

        # -1 is used as end-of-sentence char
        # FIXME: raise errors when user asks for illegal surprisal etc.
        self._vocabulary = { nonterminals[i]: i for i in range(len(nonterminals)) }
        self._vocab_r = { val: key for key, val in self._vocabulary.items() }

        self._sentences = sentences
        self._meanings = meanings

        # TODO: Move compression into sentence generation!
        self._sentences = [ tuple(self._vocabulary[word] for word in sentence )
                            for sentence in self._sentences ]

        # some sentences have variable length; append -1 as terminator digit
        sentences_evened = list(itertools.zip_longest(*self._sentences,
                                                      fillvalue=-1))
        self._sentences = np.array(sentences_evened).T
        # todo: find a faster way to sort this (condense ambiguity)
        # without constructing np.ndarray twice. see Corpus._condense_ambiguity
        self._meanings = np.array(self._meanings)

        # sort corpus
        order = np.lexsort(np.fliplr(self._sentences).T)
        self._sentences = self._sentences[order]
        self._meanings = self._meanings[order]

        self._condense_ambiguity()

    def _condense_ambiguity(self):
        """
        Checks self._sentences for repeating sentences and condenses each of
        their associated SyntaxTrees (in self._meanings) into one sentence.

        M: self._sentences, self._meanings
        :return: None
        """

        # expensive but can't think of a better solution right now
        self._meanings = [ [t] for t in self._meanings ]

        # find duplicate rows O(n), not vectorized...
        i = 0
        while i < self._sentences.shape[0] - 1:
            # todo: a better way to do this than two checks every time
            while i < self._sentences.shape[0] - 1 and \
                np.array_equal(self._sentences[i], self._sentences[i+1]):
                # condense
                self._meanings[i].append(self._meanings[i+1][0])
                del self._meanings[i+1]
                self._sentences = np.delete(self._sentences, i+1, 0)
            i += 1

        return

    @classmethod
    def from_PCFG(cls, filename):
        """
        Builds corpus from a custom PCFG in a text file.
        MODIFIES: self.grammar, self.sentences, self.meanings,
                  self.vocabulary

        :param filename: .txt in pwd
        :return: Corpus
        """

        raise NotImplementedError

    def query(self, partial, delim=' '):
        """
        Returns a CorpusSelection pointing to a subset of the Corpus._sentences
        where partial is equivalent to each sentence with at least the first
        len(partial) words. If partial is not found in corpus, returns NoneType

        :param partial: string; words to match in Corpus.
                        Must be splittable on DELIM.
        :return: CorpusSelection or NoneType
        """

        # convert partial string to query-able sentence (compress partial)
        words = partial.split(delim)
        fingerprints = tuple(self._vocabulary[word] for word in words)

        # Search Corpus._sentences for matches O(M * log N)
        scope = self._sentences
        scope_meanings = self._meanings
        for i in range(len(fingerprints)):
            lower = np.searchsorted(scope[:,i], fingerprints[i],
                                    side='left')

            # check if found
            if lower == scope.shape[0] or fingerprints[i] != scope[lower][i]:
                # no match found
                print(f"Error: no sentence matching {words[i]}"
                      f" at index {i}")
                raise KeyError(words[i])

            # narrow search
            upper = np.searchsorted(scope[:,i], fingerprints[i],
                                    side='right')
            scope = scope[lower:upper]
            scope_meanings = scope_meanings[lower:upper]

        # package CorpusSelection
        return self.CorpusSelection(words, len(words), delim, scope, scope_meanings,
                                    self._vocabulary, lower=lower, upper=upper)

    def save(self):
        """
        Saves self to a pickle binary file.

        :return: None
        """

        with open(self.name + '.corpus', 'wb') as c:
            # use default pickle protocol (3)
            pickle.dump(self, c)

        return

    def analyze_single(self):
        """
        Picks a single sentence from the Corpus and calculates all
        information metrics for each word in the sentence. Packages
        it in a pd.DataFrame, with columns for word and each metric.

        TODO: Mutual Information
        TODO: pick using actual corpus distr, not uniform

        :param corpus: corpus.Corpus
        :return: pandas.DataFrame
        """

        # needed for CorpusSelection
        DELIM = ' '

        # pick random sentence
        s_i = np.random.randint(0, high=len(self._sentences))

        # for building dataframe
        columns = ['word','surprisal','entropy_lex','entropy_struct',
                   'ER_lex', 'ER_struct', 'KL_diverge_PQ',
                   'KL_diverge_QP']
        words = list()
        surprisals = list()
        entropy_lexs = list()
        entropy_structs = list()
        ER_lexs = list()
        ER_structs = list()
        KL_diverge_PQs = list()
        KL_diverge_QPs = list()
        #mutual_infos = list()

        selection = self.CorpusSelection(list(), 0, DELIM, self._sentences,
                                         self._meanings, self._vocabulary,
                                         vocab_r=self._vocab_r,
                                         lower=0, upper=len(self._sentences))

        # judge length of sentence
        try:
            end_i = np.where(self._sentences[s_i] == -1)[0][0]
        except IndexError:
            end_i = self._sentences.shape[1]

        sentence = [ w for w in self._sentences[s_i][:end_i] ]

        for word_i in range(end_i):
            # fixme: better way to do this...?
            word = self._vocab_r[sentence[word_i]]
            words.append(word)

            # calculate them all!
            surprisals.append(selection.surprisal(word))
            entropy_lexs.append(selection.entropy_lex())
            entropy_structs.append(selection.entropy_structural())
            ER_lexs.append(selection.ER_lex(word))
            ER_structs.append(selection.ER_structural(word))
            KL_diverge_PQs.append(selection.KLdiverge_PQ(word))
            KL_diverge_QPs.append(selection.KLdiverge_QP(word))
            #mutual_infos.append(selection.mutual_info(word))

            # select for next word
            #next_word = self._vocab_r[sentence[word_i+1]]
            selection = selection.query_single(word)

        # build dataframe
        df_data = [words, surprisals, entropy_lexs,
                   entropy_structs, ER_lexs, ER_structs,
                   KL_diverge_PQs, KL_diverge_QPs]
        df_columns = { col: l for col, l in zip(columns, df_data) }
        metrics = pd.DataFrame(df_columns)
        metrics.index.name = 'i'
        metrics.columns.name = 'metric'

        return metrics

    class CorpusSelection:
        """
        A version of Corpus that calculates complexity metrics on queried.
        """
        def __init__(self, query, queried_len, delim,
                     sentences, meanings, vocab,
                     vocab_r=None, lower=None, upper=None):
            """

            :param query: ordered collection of strings
            :param queried_len: int
            :param sentences: np.ndarray of ints, sorted lexically
            :param meanings: ordered collection of ordered collections of SyntaxTree
            :param vocab: dict string to int
            :param vocab_r: dict int to string, the converse of vocab
            :param lower: int (which portion of previous CorpusSelection
                               does this one represent)
            :param upper: int (end of portion described above)
            """

            self.qlength = queried_len
            self.current_query = query
            self.delim = delim
            self._sentences = sentences
            self._meanings = meanings
            self._lower = lower
            self._upper = upper
            self._vocab = vocab

            if vocab_r:
                self._vocab_r = vocab_r
            else:
                self._vocab_r = { val: key for key, val in self._vocab.items() }

            # calculate proportion of Corpus this selection contains
            self.log_sum_probability = -np.inf
            for forest in self._meanings:
                for tree in forest:
                    self.log_sum_probability = np.logaddexp2(self.log_sum_probability,
                                                                  tree.log_likelihood)

        def _lookup(self, word):
            """
            Looks up word in vocabulary. Raises KeyError if not found.

            :param word: string
            :return: integer
            """

            try:
                fingerprint = self._vocab[word]
            except KeyError:
                # give a more relevant message
                print(f"Error: {word} not found in vocabulary")
                raise KeyError(word)

            return fingerprint

        def query_single(self, word, indices_only=False):
            """
            Returns a CorpusSelection with all sentences matching
            the current query and then 'word'.

            :param word: string (shouldn't contain self.delim)
            :param indices_only: boolean
            :return: CorpusSelection
            """

            fingerprint = self._lookup(word)
            lower = np.searchsorted(self._sentences[:,self.qlength],
                                    fingerprint, side='left')

            # check if found
            if lower == self._sentences.shape[0] or \
               fingerprint != self._sentences[lower][self.qlength]:
                # no match found
                print(f"Error: no sentence matching {word}"
                      f" at index {self.qlength}")
                raise KeyError(word)

            # narrow search
            upper = np.searchsorted(self._sentences[:,self.qlength],
                                    fingerprint, side='right')

            if indices_only: return lower, upper

            return Corpus.CorpusSelection(self.current_query + [word],
                                          self.qlength + 1, self.delim,
                                          self._sentences[lower:upper],
                                          self._meanings[lower:upper],
                                          self._vocab, vocab_r=self._vocab_r,
                                          lower=lower, upper=upper)


        def query(self, partial):
            """
            Returns a CorpusSelection with all sentences matching partial
            (up to the number of words in partial).

            :param partial: string
            :return: CorpusSelection
            """
            words = partial.split(self.delim)

            # check if partial matches all sentences in this CorpusSelection
            if len(words) <= self.qlength + 1:
                print("Error: length of new query is not greater than current"
                      "query length")
                raise KeyError(partial)
            elif words != self.current_query[:len(words)]:
                print("Error: new query does not match any sentences"
                      "in current selection")
                raise KeyError(partial)

            # recursively select next words in query
            return self.query_single(words[0])\
                   .query(partial.split(self.delim, 1))


        def surprisal(self, word):
            """
            Calculates the lexical surprisal of reading word in the context
            of self.queried. Returns None if word isn't found in CorpusSelection.
            See Hale (2016) for formula.

            :param word: string
            :return: float (surprisal)
            """

            fingerprint = self._lookup(word)
            lower = np.searchsorted(self._sentences[:,self.qlength],
                                    fingerprint, side='left')

            # check if word doesn't appear after context
            if lower == self._sentences.shape[0] or \
               self._sentences[lower, self.qlength] != fingerprint:
                # surprisal is very high....
                # TODO: surprisal for unseen word?
                return np.inf

            # calculate surprisal
            upper = np.searchsorted(self._sentences[:,self.qlength],
                                    fingerprint, side='right')

            log_prob_word = -np.inf # "log2(0)"
            for i in range(lower, upper):
                for tree in self._meanings[i]:
                    log_prob_word = np.logaddexp2(log_prob_word,
                                                  tree.log_likelihood)

            surprisal = -(log_prob_word - self.log_sum_probability)

            # for some reason -0.0 is a valid float
            return 0 if surprisal == 0 else surprisal

        def entropy_lex(self):
            """
            Calculates the lexical entropy after reading self.queried.
            See Hale (2016) for formula.

            :return:  float (entropy)
            """

            # check if we're done reading the sentence
            if len(self._sentences) == 1:
                return 0

            # get list of all next possible words
            u, unique_indices, unique_counts = np.unique(self._sentences[:,self.qlength],
                                                         return_index=True,
                                                         return_counts=True)

            entropy = 0
            for index, count in zip(unique_indices, unique_counts):
                # calculate summation term
                # log2( P(word | context) )
                logPword_g_select = -np.inf # to represent log2(0)
                for forest in self._meanings[index:index+count]:
                    # generator of each log2( P(meaning | context) )
                    logPtree_g_selects = (tree.log_likelihood - self.log_sum_probability
                                         for tree in forest)
                    # sum over meanings to get log2( P(word | context) )
                    for logPtree_g_select in logPtree_g_selects:
                        logPword_g_select = np.logaddexp2(logPword_g_select,
                                                          logPtree_g_select)

                # formula for entropy
                term = (2 ** logPword_g_select) * logPword_g_select
                entropy += term

            return -entropy if entropy != 0 else 0

        def entropy_structural(self):
            """
            Calculates the entropy over possible meanings
            after reading self.queried.
            See Hale (2016)

            :return: float (meaning entropy)
            """

            # Every meaning (SyntaxTree) in CorpusSelection
            # is a term in Structural Entropy
            entropy = 0
            for forest in self._meanings:
                # generator of each log2( P(meaning | context) )
                logPtree_g_selects = (tree.log_likelihood - self.log_sum_probability
                                      for tree in forest)
                for logPtree_g_select in logPtree_g_selects:
                    # formula for entropy
                    term = (2 ** logPtree_g_select) * logPtree_g_select
                    entropy += term

            # don't return -0.0
            return -entropy if entropy != 0 else 0

        def ER_lex(self, word):
            """
            Calculates the lexical entropy reduction after reading word in the
            context of self.queried.

            :param word: string; in self.vocabulary
            :return: float (lexical entropy reduction)
            """

            current_entropy = self.entropy_lex()
            entropy_reduction = current_entropy \
                                - self.query_single(word).entropy_lex()

            # see Hale (2016) #FIXME: correct paper?
            return max(entropy_reduction, 0)

        def ER_structural(self, word):
            """
            Calculates the entropy reduction in possible meanings
            after reading word in the context of self.queried.

            :param word:  string; in self.vocabulary
            :return:  float (structural entropy reduction)
            """

            entropy_reduction = self.entropy_structural() \
                                - self.query_single(word).entropy_structural()

            return max(entropy_reduction, 0)

        def KLdiverge_PQ(self, word):
            """
            Dkl (Pk+1 || Pk) = ∑ q(T) log (q(T) / p(T))
            where Q = Pk+1 and P = Pk

            Calculates the Kulback-Leibler Divergence from the
            distribution over currently possible meanings (Q)
            to the distr over possible meanings after reading "word" (P).

            Note: Levy (2008) has shown that this value is equivalent to
            the lexical suprisal (Hale, 2001) of the next word given
            the current context. If these numbers aren't equivalent, then
            that's an issue!

            :return: float (KL divergence)
            """

            # we could just return surprisal, but that would be too easy...
            next_selection = self.query_single(word)

            # ignore all terms where q(T) is zero (outside of next_word CorpusSelection)
            # effectively, assuming 0 * log (0) = 0
            divergence = 0
            for forest in next_selection._meanings:
                for tree in forest:
                    logq = tree.log_likelihood - next_selection.log_sum_probability
                    logp = tree.log_likelihood - self.log_sum_probability

                    # rewrote formula for log-probabilities
                    divergence += (2 ** logq) * (logq - logp)

            return divergence

        def KLdiverge_QP(self, word):
            """
            Dkl (Pk || Pk+1) = ∑ p(T) log (p(T) / q(T))
            where Q = Pk+1 and P = Pk; T = a meaning

            Calculates the Kulback-Leibler Divergence from the
            distribution over possible meanings after reading
            "word" (P) to the distr over currently possible meanings (Q).

            :param word: string; R: len(string.split(self.delim)) == 1
            :return: float (KL divergence)
            """

            next_selection = self.query_single(word)

            if not (next_selection._lower and next_selection._upper):
                next_selection._lower, next_selection._upper = self.query_single(word,
                                                                                 indices_only=True)

            divergence = 0
            for forest in itertools.chain(self._meanings[:next_selection._lower],
                                          self._meanings[next_selection._upper:]):
                for tree in forest:
                    actualq = 0 # not in next_selection
                    logp = tree.log_likelihood - self.log_sum_probability

                    # rewrote formula for log-probabilities
                    # FIXME: what to do with log2(0) in this situation?
                    divergence += (2 ** logp) * (logp - np.log2(actualq))

            for forest in self._meanings[next_selection._lower:next_selection._upper]:
                for tree in forest:
                    logq = tree.log_likelihood - next_selection.log_sum_probability
                    logp = tree.log_likelihood - self.log_sum_probability

                    divergence += (2 ** logp) * (logp - logq)

            return divergence

        def mutual_info(self):
            """
            Calculates Mutual Information metric between the distr over
            possible meanings in current context and the distr over
            possible meanings after reading the next word.

            formula: ∑ ∑ p(x ^ y) * log2 ( p(x ^ y) / p(x) * p(y) )
                    summed pairwise over all x and y
                where x = self.current_query, y = self.current_query + word

            :param word: string
            :return: float (Mutual Information)
            """



            raise NotImplementedError
