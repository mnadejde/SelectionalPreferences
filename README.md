Selectional Preferences Feature
===============================


Scripts for computing different models for a Selectional Prefrences feature described in this paper:

Nadejde, Maria  and  Birch, Alexandra  and  Koehn, Philipp (2016): Modeling Selectional Preferences of Verbs and Nouns in String-to-Tree Machine Translation. Proceedings of the First Conference on Machine Translation. Berlin, Germany

https://aclweb.org/anthology/W/W16/W16-2204.pdf

Moses features
--------------
https://github.com/moses-smt/mosesdecoder/blob/maria_SelPref/moses/FF/SelPrefFeature.cpp

https://github.com/moses-smt/mosesdecoder/blob/maria_SelPref/moses/FF/SelPrefFeature.h

        
Models
------

WittenBellModel.py:


Estimate P(arg|vb,rel) with Witten Bell smoothing

Input: (verb, dependency relation, argument) tuples, one per line, extracted from parsed text.

Output: Model written in ARPA format # PWB(arg|rel vb) rel vb arg BOW(rel, vb). To be queried in Moses with KenLM.


SelectionalAssociationModel.py:


Computes direct and inverse selectional preference and selectional association scores according to Resnik and Erk&Pado (A flexible, Corpus-Driven Model of selectional preference)

SelPrefStrength(v,r) = KL(P(arg|v,r)||P(arg|r)) = sum_over_arg (P(arg|v,r) * log( P(arg|v,r)/P(arg|r) )

SelAssoc(v,r,arg) = 1/SelPrefStrength(v,r) * P(arg|v,r) * log( (P(arg|v,r)/P(arg|r) )

SelPrefStregthInverse(arg,r) = KL(P(v|arg,r)||P(v|r)) = sum_over_v P(v|arg,r) * log( (P(v|arg,r)/P(v|r) )

SelAssocInverse(arg,r,v) = 1/SelPrefStrengthInverse(arg,r) * P(v|arg,r) * log((P(v|arg,r)/P(v|r))


Computes a mutual information based model: sigmoid(P(arg|rel,vb) / (P(arg|rel))

Has option to load a dictionary from words to classes to be used as arguments.




