"""
artificial_grammar.py

Language and Cognitive Architecture Lab at University of Michigan
Sean Anderson, Research Assistant

7/2/19

Implements a Probablistic, Context-Free Grammar, custom specified with a text
file, that generates legal sentences. Can limit depth of sentences, making
it possibly Finite-State. Used for building a Corpus (see corpus.py)
"""

import numpy as np # for random numbers
from math import log2
import sys
from corpus import Corpus, SyntaxTree

class Rule:
    """
    Custom object to represent a probabalistic
    phrase structure rule.
    """
    def __init__(self, probability, name, expression):
        self.exp_distr = list([probability])
        self.parent = name
        self.exp = list([expression])

    def _check_recursion(self):
        """
        R: self.exp_distr, parent, exp defined
        M: self.has_recurs_exp, self.n_r_index
        E: Determines if self contains recursive realizations.
           Assigns an index to a non-recursive expression in self.exp.
        """

        # track which exp's are recursive, and whether rule can be recursive
        self.has_recurs_exp = False
        self.recurs_exps = [ False for i in self.exp ]
        self.n_r_index = len(self.exp)

        for i in range(len(self.exp)):
            if self.parent in self.exp[i]:
                self.has_recurs_exp = True
                self.recurs_exps[i] = True
            else:
                self.n_r_index = i

        # didn't find non-recursive expression for rule
        if self.has_recurs_exp and self.n_r_index == len(self.exp):
            print("WARNING: Recursive rule has no non-recursive"
                  "realization. Cannot limit generation depth."
                  " May result in crash during generation")

        return self.has_recurs_exp

    def generate(self):
        """
        R: len(exp_distr) = len(exp)
           sum(exp_distr) == 1.0
        E: Returns one of the expressions with the probability
           described in exp_distr
        """
        choice = np.random.choice(len(self.exp), p=self.exp_distr)
        # FIXME: max tree depth among different rules?
        return self.exp[choice]

    def __str__(self):
        return (f"parent: {self.parent}\n"
                f"distr: {self.exp_distr}\nexp's: {self.exp}")

    def __repr__(self):
        # Assumes Rule is legal
        s = '['
        for i in range(len(self.exp)):
            s += f"{self.exp_distr[i]} {self.parent}"\
                 + f" -> {self.exp[i]}; "
        return s[:-2] + ']'

class Grammar:
    def __init__(self, filename, initial, max_depth=2, limit_depth=True,
                 delim=' '):
        """
        Initializes Grammar using custom grammar

        :param filename: .txt file in pwd
        :param initial: string; parent symbol to start derivations
        :param max_depth: int
        :param limit_depth: boolean
        :param delim: string (word separator)
        """

        # FIXME: move INITIAL declaration to .txt file
        self.INITIAL = initial
        self.DELIM = delim
        self.max_depth = max_depth
        self.limit_depth = limit_depth
        self.is_recursive = False # default assumption
        self.rules = dict()
        self.name = filename # for pickle save
        self._init_from_file(filename)

    def _init_from_file(self, filename, print_rules=True):
        """
        R: filename is in pwd
           filename is a plain text file
        M: self
        E: Constructs self.rules from specifications in text file.

        Text file requirements:
         - lines starting with # denote comments and are ignored.
         - Other lines are interpreted as phrase rules and should be
           formatted as so:
           Probability PhraseName '->' ConstituentA ConstiuentB ...
         - All phrase rules with the same PhraseName should have
           Probabilities that add up to 1.0. Otherwise rule construction
           will fail.
         - At least one phrase rule must have the name 'CP'. This will
           be the starting point for generation.
         - Currently, only one recursive rule (rule that can expand to
           include itself) is allowed to enable depth limiting.
         - Do not define multiple 'abstract' rules (i.e. rules like CP
           that don't exclusively express 'surface' rules like S and R).
           This avoids the situation where CP can express XP and XP can
           express CP, which prevents the forcing of surface-type
           symbol generation for depth limiting. I'm still working on
           a surefire way to enforce this upon initialization.
        """
        self.surfaces = set()

        with open(filename) as f:
            for line in f:
                # skip comment lines
                if line[0] == '#' or line[0] == '\n': continue

                # parse line into a phrase rule
                phrase = line.split()
                exp = tuple(phrase[3:])
                name = phrase[1]
                probability = float(phrase[0])

                # used by self.check_grammar
                self.surfaces.update(phrase[3:])

                # check if belongs to existing rule
                if name in self.rules:
                    # add to existing rule
                    current = self.rules[name]
                    current.exp_distr.append(probability)
                    current.exp.append(exp)
                else:
                    # create new rule
                    current = Rule(probability, name, exp)
                    self.rules[name] = current
        # file closed

        self.surfaces -= self.rules.keys()

        if not self.check_grammar():
            # FIXME: come on, be Pythonic...
            raise GrammarError

        # confirm rules read in correctly
        if print_rules:
            print("Rules\n=====")
            for rule in self.rules.values():
                print(rule)
                print('')
            print('')

        return

    def check_grammar(self):
        """
        R: !self.rules.empty(), self.surfaces exists
        E: Returns false if probability constraints of phrase rules
           are violated. Otherwise returns true.
        """
        EPSILON = 0.001 # for float comparison
        is_abstract = False

        for rule in self.rules.values():
            if (abs(sum(rule.exp_distr) - 1.0) > EPSILON
                or len(rule.exp_distr) != len(rule.exp)):
                   # faulty rule!
                   print("Error: Illegal grammar: see rule")
                   print(str(rule))
                   return False

            # Enforce qualifications for depth-limited generation
            if rule._check_recursion():
                self.is_recursive = True
                if rule.n_r_index == -1:
                    print("Error: Recursive rule does not have "
                          "non-recursive realization")
                    print(rule)
                    return False
            # Enforce only one abstract rule
            # TODO: need a better way to do this
            #for e in rule.exp:
            #    for w in e:
            #        if w not in self.surfaces:
            #            if not is_abstract:
            #                is_abstract = True
            #            else:
            #                print("Error: More than one abstract rule"
            #                      " (prevents depth limit)")
            #                print("FIXME: test for abstract rules FAULTY!"
            #                      " ignore")

        return True

    def generate_corpus(self):
        """
        R: (for custom grammar generation)
           self.rules exists
        E: Generates all legal sentences using provided Rules and returns a
           Corpus object with each sentence and its meaning (stored as SyntaxTree).
           Uses a left-branching Backtracking strategy.
        """

        # FIXME: toggle depth limiting here
        self.sentences, self.likelihoods = list(), list()

        # start deriving! log2(1) = 0
        if self.limit_depth:
            deriv = [(self.INITIAL, 0)]
            self._gen_corpus_limited(deriv, 0, list())
        else:
            # check if recursive grammar
            if self.is_recursive:
                print("Warning: no depth-limit specified when using recursive grammar."
                      " Please specify a limit")
                return None

            deriv = [self.INITIAL]
            self._gen_corpus_helper(deriv, 0, list())

        return self._package_corpus()

    def _gen_corpus_helper(self, deriv, likelihood, sentence):
        """
        Expands the next rule parent (nonterminal) in the derivation
        using a left-first, DFS Backtracking strategy. Upon reaching a complete
        sentence, adds sentence and its likelihood to self and returns.

        WARNING: FIXME: No Depth limiting. Will run indefinitely
        with recursive grammars.

        :param deriv: list of strings, used as a stack. Each string in self.rules
        :param likelihood: float (log P(deriv | Grammar))
        :param sentence: list of strings, each string in self.surfaces
        :return: None
        """

        # base case: solution found
        if not deriv: # empty
            self.sentences.append(sentence.copy())
            self.likelihoods.append(likelihood)
            return
        # normal case: expand next possible derivations
        else:
            parent = deriv.pop()
            curr_rule = self.rules[parent]

            # FIXME: duplicating stacks recursively costs lots of unnecessary memory!
            # FIXME: and time too!

            for exp_i in range(len(curr_rule.exp)):
                next_deriv = deriv.copy()
                next_sentence = sentence.copy()

                # add nonterminals to stack in reverse order
                num_terminals = 0
                num_nonterminals = 0
                for item in reversed(curr_rule.exp[exp_i]):
                    if item in self.surfaces:
                        # terminal
                        next_sentence.append(item)
                        num_terminals += 1
                    else:
                        next_deriv.append(item)
                        num_nonterminals += 1

                new_likelihood = likelihood + log2(curr_rule.exp_distr[exp_i])
                self._gen_corpus_helper(next_deriv, new_likelihood, next_sentence)

                # remove expansion for next branch O(max E length)
                # 'del' to ensure modifying parent stack's name
                #del deriv[-num_nonterminals:]
                #del sentence[-num_terminals:]

            # add parent back for upper recursion to continue searching...
            #deriv.append(curr_rule.parent)

            # remove exhausted nonterminal (parent) to avoid over-generation
            #deriv.pop()

        return

    def _gen_corpus_limited(self, deriv, likelihood, sentence):
        """
        Expands the next nonterminal symbol in deriv. Uses a left-first,
        Backtracking (DFS) strategy. Forces non-recursive symbol transformations
        when approaching maximum depth. Note: this violates the exact
        probabilistic distributions specified in the grammar rules.

        :param deriv: list of strings, used as a stack
        :param likelihood: float
        :param sentence: list of strings
        :return: Nothing
        """

        # check if complete
        if not deriv:
            self.sentences.append(sentence.copy())
            self.likelihoods.append(likelihood)
            return

        # not finished; expand next node
        parent, recurs_depth = deriv.pop()
        curr_rule = self.rules[parent]

        # FIXME: duplicating stacks recursively costs lots of unnecessary memory!
        # FIXME: and time too!

        if recurs_depth >= self.max_depth - 1 and \
           curr_rule.has_recurs_exp:
            # force non-recursive transformation
            expressions_i = [curr_rule.n_r_index]
        else:
            # transform normally, try all
            expressions_i = range(len(curr_rule.exp))

        # Transform parent symbol
        for i in expressions_i:
            next_deriv = deriv.copy()
            next_sentence = sentence.copy()

            # add nonterminals to stack in reverse order
            num_terminals = 0
            num_nonterminals = 0
            for symbol in reversed(curr_rule.exp[i]):
                if symbol in self.surfaces:
                    # terminal
                    # FIXME: use compression after debugging
                    next_sentence.append(symbol)
                    num_terminals += 1
                else:
                    next_depth = recurs_depth if not curr_rule.recurs_exps[i] \
                                              else recurs_depth + 1
                    next_deriv.append((symbol, next_depth))
                    num_nonterminals += 1

            new_likelihood = likelihood + log2(curr_rule.exp_distr[i])
            self._gen_corpus_limited(next_deriv, new_likelihood, next_sentence)

    def _package_corpus(self):
        """
        R: self.sentences and self.meanings exist; (both are coindexed)

        :return: corpus.Corpus
        """

        # package meanings
        # renormalize if probabilities altered (use optimized numpy code)
        # FIXME: losing some precision here...
        if self.limit_depth:
            likelihoods = np.array(self.likelihoods)
            likelihoods = (2 ** likelihoods) / (2 ** likelihoods).sum()
            meanings = [ SyntaxTree(np.log2(p)) for p in likelihoods ]
        else:
            meanings = [ SyntaxTree(p) for p in self.likelihoods ]

        return Corpus(tuple(self.surfaces), self.sentences,
                      meanings, self.name[:-4])

def test(filename):
    grammar = Grammar(filename, initial='CP', limit_depth=True,
                      delim=' - ', max_depth=2)
    grammar.generate_corpus()

    actual_likelihoods = list()
    for i in range(len(grammar.sentences)):
        print(grammar.sentences[i], 2 ** grammar.likelihoods[i])
        actual_likelihoods.append(2 ** grammar.likelihoods[i])


    print(f"should add to 1: {sum(actual_likelihoods)}")

    return 0

if __name__ == '__main__':
    test(sys.argv[1])
