"""
pcfg.py

7/11/19

Sean Anderson, Research Assistant
Language and Cognitive Architecture Lab, University of Michigan

Probabalistic Context-Free Grammar simulator. Reads a custom grammar
and builds a Corpus of all legal sentences. Simulates sentence
generation while providing 8 information-theoretic metrics.
"""

from corpus import Corpus
from artificial_grammar import Grammar
import argparse
import pickle
import numpy as np
import pandas as pd

def set_parser():
    # Initiates command-line parser for the program.
    DESCRIPTION = "Simulate sentence generation from a Custom PCFG," \
                  "with Information Theoretic Metrics"
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser.add_argument("grammar_file", help="custom grammar.txt file")
    parser.add_argument("initial_symbol", help="symbol used to start generation")

    parser.add_argument("-l", "--limit_depth",
                        help="set depth limit for recursive generation",
                        type=int)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-p", "--pickle",
                       help="use already existing Corpus pickle file",
                       action='store_true')
    group.add_argument("-g", "--generate",
                       help="force generation of Corpus using grammar",
                       action='store_true')

    parser.add_argument("-s", "--single",
                        help="generate only one sentence",
                        action='store_true')

    return parser

def quick_test(corpus):
    # test basic corpus functionality
    selection = corpus.query('Circle LeftOf', ' ')
    print(f"Circle LeftOf... Triangle surprisal: "
          f"{selection.surprisal('Triangle')}")
    print(f"Circle LeftOf... Triangle KL divergence (P || Q): "
          f"{selection.KLdiverge_PQ('Triangle')}")
    print(f"Circle LeftOf... lexical entropy: "
          f"{selection.entropy_lex()}")
    print(f"Circle LeftOf... structural entropy: "
          f"{selection.entropy_structural()}")
    print(f"Circle LeftOf... Triangle Entropy Reduction (lexical): "
          f"{selection.ER_lex('Triangle')}")
    print(f"Circle LeftOf... Triangle Entropy Reduction (structural): "
          f"{selection.ER_structural('Triangle')}")
    print(f"Circle LeftOf... Triangle KL divergence (Q || P): "
          f"{selection.KLdiverge_QP('Triangle')}")


    selection = corpus.query('Square')
    print(f"Square... Above surprisal: "
          f"{selection.surprisal('Above')}")
    print(f"Square... Above KL divergence (P || Q): "
          f"{selection.KLdiverge_PQ('Above')}")
    print(f"Square... lexical entropy: "
          f"{selection.entropy_lex()}")
    print(f"Square... structural entropy: "
          f"{selection.entropy_structural()}")
    print(f"Square... Above Entropy Reduction (lexical): "
          f"{selection.ER_lex('Above')}")
    print(f"Square... Above Entropy Reduction (structural): "
          f"{selection.ER_structural('Above')}")
    print(f"Square... Above KL divergence (Q || P): "
          f"{selection.KLdiverge_QP('Above')}")

    selection = corpus.query('Diamond Above')
    print(f"Diamond Above... Diamond surprisal: "
          f"{selection.surprisal('Diamond')}")
    print(f"Diamond Above... Diamond KL divergence (P || Q): "
          f"{selection.KLdiverge_PQ('Diamond')}")
    print(f"Diamond Above... lexical entropy: "
          f"{selection.entropy_lex()}")
    print(f"Diamond Above... structural entropy: "
          f"{selection.entropy_structural()}")
    print(f"Diamond Above... Diamond Entropy Reduction (lexical): "
          f"{selection.ER_lex('Diamond')}")
    print(f"Diamond Above... Diamond Entropy Reduction (structural): "
          f"{selection.ER_structural('Diamond')}")
    print(f"Diamond Above... Diamond KL divergence (Q || P): "
          f"{selection.KLdiverge_QP('Diamond')}")

    selection = corpus.query('Diamond')
    print(f"Diamond... Square surprisal: "
          f"{selection.surprisal('Square')}")
    print(f"Diamond... Square KL divergence (P || Q): "
          f"{selection.KLdiverge_PQ('Square')}")
    print(f"Diamond... Square Entropy Reduction (lexical)"
          f"{selection.ER_lex('Square')}")
    print(f"Diamond... Square KL divergence (Q || P): "
          f"{selection.KLdiverge_QP('Square')}")


    try:
        selection = corpus.query('Below')
    except KeyError:
        print('successfully terminated on wrong word')

    return 0

def analyze_all(corpus):
    """
    Writes all sentences in corpus and all the complexity metrics
    at each word to a csv. The format is as follows:
    :param corpus:
    :return:
    """

    raise NotImplementedError

def main():
    parser = set_parser()
    args = parser.parse_args()

    # check if corpus already saved as pickle
    if args.pickle:
        # TODO: search all .corpus binary files in directory,
        # and select the one that matches the grammar file.

        # for now, just check if there's a corpus matching
        # the grammar file supplied
        corpus_name = args.grammar_file[:-3] + 'corpus'
        try:
            c = open(corpus_name, 'rb')
        except IOError:
            print(f"Error: No matching corpus save for {args.grammar_file}")
            return 0
        with c:
            corpus = pickle.load(c)
            print(f"Found matching {corpus_name}. Using...")
    else:
        # read grammar
        grammar = Grammar(args.grammar_file, args.initial_symbol,
                          limit_depth=args.limit_depth,
                          max_depth=args.limit_depth)
        corpus = grammar.generate_corpus()

    # TODO: option for saving
    corpus.save()

    # TODO: create well-formatted csv with all data...?
    if args.single:
        # print single sentence with complexity metrics
        analysis = corpus.analyze_single()
        print(analysis)
        analysis.to_csv('sentence.csv')
    else:
        analyze_all(corpus)

    return 0

if __name__ == '__main__':
    main()
