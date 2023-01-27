"""
test.py

draft for Grammar.generate_corpus
"""

def Grammar.generate_corpus(self):
    """
    Unlimited Depth, EXPAND FIRST version
    WARNING: will run indefinitely with any recursive grammar
    """
    self.sentences = list()
    self.meanings = list()

    next_parent = self.INITIAL # set at construction
    next_exp = 0

    next_rule = self.rules[next_parent]

    for exp_i in range(len(next_rule.exp)):
        self.generate_corpus_helper(list(), next_parent, exp_i,
                                    1, list())
    return

def Grammar.generate_corpus_helper(self, deriv, likelihood, sentence):
    """
    deriv: list of strings (rule parents)
    likelihood: float (multiplied probabilities of each expansion)
    sentence: list of strings, each in self.surfaces
    """
    # base case: solution found
    if !deriv: # empty
        # sentence complete
        self.sentences.append(sentence)
        self.likelihoods.append(likelihood)
        return
    # normal case: run through next possible derivations
    else:
        current = deriv.pop()
        curr_rule = self.rules[current]

        # run through possible expansions
        for exp_i in range(len(curr_rule.exp)):
            # add nonterminals to stack in reverse order
            for item in reversed(curr_rule.exp[exp_i]):
                if item in self.surfaces:
                    # terminal (surface word)
                    sentence.append(item)
                else:
                    deriv.append(item)

            new_likelihood = likelihood * curr_rule.exp_distr[exp_i]
            self.generate_corpus_helper(deriv, new_likelihood, sentence)

            # remove new expression from stack for next expansions O(E)
            deriv = deriv[:-len(curr_rule.exp[exp_i])]

    return

def Grammar.generate_corpus_helper(self, deriv, next_parent, next_exp
                                   likelihood, sentence):
    """
    EXPAND FIRST, ASK QUESTIONS LATER version (faster?)

    next_parent: string, to-be-expanded parent
    next_exp: int, to-be-expanded expression of next_parent
    deriv: list of strings (rule parents) used as stack
    likelihood: float (multiplied probabilities of each expansion)
    sentence: list of strings, each in self.surfaces (terminals)
    """
    # expand next
    curr_rule = self.rules[next_parent]
    new_likelihood = likelihood * curr_rule.exp_distr[next_exp]

    # add nonterminals to stack in reverse order
    for item in reversed(curr_rule.exp[next_exp]):
        if item in self.surfaces:
            # terminal (surface word)
            sentence.append(item)
        else:
            deriv.append(item)

    # base case: solution found
    if !deriv: # empty
        # sentence complete
        self.sentences.append(sentence)
        self.likelihoods.append(new_likelihood)
        return
    # normal case: run through next possible derivations
    else:
        top_rule = deriv.pop()
        for exp_i in range(len(top_rule.exp)):
            self.generate_corpus_helper(deriv, top_rule, exp_i,
                                        likelihood, sentence)

    return



CP
S R S (0.333)
Triangle R S (0.333 * 0.25)
Triangle LeftOf S (0.333 * 0.25 * 0.25)
Triangle LeftOf Square (0.333 * 0.25 * 0.25)

CP
S R CP
Triangle R CP
Triangle LeftOf CP
Triangle LeftOf S R S
Triangle LeftOf Square R S
Triangle LeftOf Square Above
Triangle LeftOf Square Above Diamond (0.333^2 * 0.25^5)
