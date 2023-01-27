# PCFG generator

This tool inputs a custom probabilistic context-free grammar (PCFG) and
computes information-theoretic metrics (e.g. suprisal) within a
sentence. Originally crafted to support projects such as [(Cho & Lewis, 2019)](https://aclanthology.org/W19-2906).

See ```PCFG_Simulator_Tutorial.ipynb``` for more details. Currently the tool does
not handle grammars with mutual recursion and there is an issue with calculating
the KL divergence.

