import fileinput
import sys
import re
import itertools
import math
import numpy as np
import operator
import cStringIO

from Util import filter_vb
from Util import filter_arg
from Util import increment_dict
from Util import print_dict_file2
from Util import print_dict_file3
from Util import print_dicts_file3
from Util import print_combine

import nltk
# module for computing a Conditional Frequency Distribution
from nltk.probability import ConditionalFreqDist

# module for computing a Conditional Probability Distribution
from nltk.probability import ConditionalProbDist

# module for computing a probability distribution with the Maximum Likelihood Estimate
from nltk.probability import MLEProbDist

# module for computing a Frequency Distribution
from nltk.probability import FreqDist

import argparse

##################
## Estimate P(arg|vb,rel)
## Input: file with tuples (rel,vb,arg), one on each line, extracted using StanfordDependencies from the parsed side of the parlalle corpus
## arg and vb are words and rel can be one of nsubj,nsubjpas, pobj etc.
## Computes direct and inverse selectional preference and selectional association scores according to Resnik and Erk&Pado (A flexible, Corpus-Driven Model of selectional preference)
## First version does not use clusters for arguments, but the lexical item directly -> probably sparse model
## Model:
## SelStr(w1,r) = KL(P(w2|w1,r)||P(w2|r)
## SelAssoc(w1,r,w2) = 1/SelStr(w1,r) * P(w2|w1,r) * log( P(w2|w1,r)/P(w2,r) )
## Direct model: w1 = verb lemma, w2 = noun lemma
## Inverse model: w1 = nount lemma, w2 = verb lemma
## Output:
# SelPrefStrength(v,r) = KL(P(arg|v,r)||P(arg|r)) = sum_over_arg (P(arg|v,r) * log( P(arg|v,r)/P(arg|r) )
# SelAssoc(v,r,arg) = 1/SelPrefStrength(v,r) * P(arg|v,r) * log( (P(arg|v,r)/P(arg|r) )
# Inverse measures:
# SelPrefStregthInverse(arg,r) = KL(P(v|arg,r)||P(v|r)) = sum_over_v P(v|arg,r) * log( (P(v|arg,r)/P(v|r) )
# SelAssocInverse(arg,r,v) = 1/SelPrefStrengthInverse(arg,r) * P(v|arg,r) * log((P(v|arg,r)/P(v|r))

#we need 4 Conditional Frequency Distributions
# Direct model
# F(arg|vb,rel)
argVbRelCFD=ConditionalFreqDist()
# F(arg|rel)
argRelCFD=ConditionalFreqDist()
# Inverse model
# F(vb|arg,rel)
vbArgRelCFD=ConditionalFreqDist()
# F(vb|rel)
vbRelCFD=ConditionalFreqDist()

lemma_dict = {}
vbDict = {}
argDict = {}
w2c = {}


def ReadDictFromFile(fileName,dict):
  f = open(fileName,'r')
  line = f.readline().split()
  while line and len(line)>1:
    #I have problems with encoding when I create the lemma file
    dict[str(line[0]).strip().lower()]=str(line[1]).strip().lower()
    #               dict[str(line[0]).strip().encode('utf-8').lower()]=str(line[1]).strip().encode('utf-8').lower()
    line = f.readline().split()

def lemmatize(arg):
  if lemma_dict.has_key(arg.lower()):
    #print "has lemma: ",arg, lemma_dict[arg]
    return lemma_dict[arg.lower()]
  else:
    return arg

def ProcessArgVb(arg,vb,w2c=""):
  # no dictionary available mapping words to clusters -> default processing is lemmatizing and filtering
  if not w2c:
    # lemmatize
    arg_l = lemmatize(arg)
    vb_l = lemmatize(vb)
    arg_f = filter_arg(arg_l)
    return arg_f, vb_l, True
  else:
    # lowercase verb and argument; map lowercased argument to it's class using w2c dictionary
    vb_p = vb.lower()
    if w2c.has_key(arg.lower()):
      arg_p = w2c[arg.lower()]
      return arg_p, vb_p, True
    else:
      arg_l = lemmatize(arg)
      if w2c.has_key(arg_l):
        arg_p = w2c[arg_l]
        return arg_p, vb_p, True
      else:
        return filter_arg(arg_l), vb_p, False

def ComputeFreqDist(f,w2c):
  global argVbRelCFD, argRelCFD, vbArgRelCFD, vbRelCFD, lemma_dict, vbDict, argDict
  incompletePairs=0
  notInW2C=0
  line = f.readline()
  while(line):
    tokens = line.split()
    if(len(tokens)>2):
      rel=tokens[0]
      vb=tokens[1]
      arg=tokens[2]
      if filter_vb(vb):
        arg_p, vb_p, found = ProcessArgVb(arg,vb,w2c)
        if (found == False):
				  notInW2C += 1
        increment_dict(vbDict, vb_p, 1)
        increment_dict(argDict, arg_p, 1)
        # Direct model
        # F(arg|vb,rel)
        argVbRelCFD[(vb_p, rel)][arg_p]+=1
        # F(arg|rel)
        argRelCFD[rel][arg_p]+=1
        # Inverse model
        # F(vb|arg,rel)
        vbArgRelCFD[(arg_p, rel)][vb_p]+=1
        # F(vb|rel)
        vbRelCFD[rel][vb_p]+=1
    else:
      incompletePairs+=1
    line = f.readline()
  return incompletePairs, notInW2C

def ComputeKLDivergence(P_condition,P_base,CFD_condition,w1Dict,w2Dict):
  # P_condition = P(w2|w1,r)
  # P_based = P(w2|r)
  SelStr = {}
  for (w1,r) in CFD_condition.conditions():
    if(w1Dict[w1]>50):
      # SHOULD IT BE w2??
      SelStr[(w1,r)] = sum([P_condition[(w1,r)].prob(w2[0]) * (math.log(P_condition[(w1,r)].prob(w2[0])+0.0000001) - math.log(P_base[r].prob(w2[0])+0.0000001)) for w2 in CFD_condition[(w1,r)].items() if w2Dict[w2[0]]>50 and w2[1]>5 ])
  return SelStr

def ComputeSelAssoc(P_condition,P_base,CFD_condition,SelStr, w1Dict,w2Dict):
  # P_condition = P(w2|w1,r)
  # P_base = P(w2|r)
  SelAssoc = {}
  for (w1,r) in CFD_condition.conditions():
    for w2 in CFD_condition[(w1,r)].items(): #(w2,freq)
      if(w1Dict[w1]>50 and w2Dict[w2[0]]>50 and SelStr[(w1,r)]!=0 and w2[1] >5):
        p_cond = P_condition[(w1,r)].prob(w2[0])+0.0000001
        p_base = P_base[r].prob(w2[0])+0.0000001
        # SelAssoc(v,r,arg) = 1/SelStr(v,r) * P(arg|v,r) * log( (P(arg|v,r)/P(arg|r) )
        SelAssoc[(w1,r,w2[0])] = ( p_cond * (math.log(p_cond) - math.log(p_base))) / SelStr[(w1,r)]
  return SelAssoc

# sigma(PMI(arg,vb,rel)) = P(arg|rel,vb) / (P(arg|rel,vb)+P(arg|rel))
# sigma(x) = 1/ (1 + e^-x)
# PMI(x,y) = p(x,y)/(P(x)*P(y)) = P(x|y)/P(x)


def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def ComputeSigmaPMI(P_condition,P_base,CFD_condition,w1Dict,w2Dict):
  SigmaPMI = {}
  # P_condition = P(w2|w1,r)
  # P_based = P(w2|r)
  for (w1,r) in CFD_condition.conditions():
    for w2 in CFD_condition[(w1,r)].items():
      if(w1Dict[w1]>50 and w2Dict[w2[0]]>50 and w2[1] >5) and P_base[r].prob(w2[0]) > 0.0000001:
        pmi = P_condition[(w1,r)].prob(w2[0]) / P_base[r].prob(w2[0])
        SigmaPMI[(r,w1,w2[0])] = sigmoid(pmi)
  return SigmaPMI

def main():
  global argVbRelCFD, argRelCFD, vbArgRelCFD, vbRelCFD, lemma_dict, vbDict, argDict, w2c
  parser = argparse.ArgumentParser()
  parser.add_argument('--deprel_file', default='../corpus.5.StanfordDep.en.100K', type=str)
  parser.add_argument('--lemma_dict_file', default='../word_lemma.pairs', type=str)
  parser.add_argument('--w2c_dict_file', default=None, type=str) #default='w2c.txt500'
  parser.add_argument('--out_file_prefix', default='../corpus.5.StanfordDep.test2.MI_model', type=str)
  parser.add_argument('--sigma_pmi', default = True, type = bool)
  parser.add_argument('--sel_assoc', default = False, type = bool)
  parser.add_argument('--print_all', default = True, type = bool)
  parser.add_argument('--print_combine', default = False, type = bool)
  args = parser.parse_args()
  
  
  depRelFile=open(args.deprel_file,'r')  #file with dep rel tuples
  #file for w2c dictionary mapping words to clusters
  if (args.w2c_dict_file is not None):
		ReadDictFromFile(args.w2c_dict_file, w2c)
		print "--- Loaded word-to-cluster dict ---"
		print "dict size: ", len(w2c.items())
  ReadDictFromFile(args.lemma_dict_file,lemma_dict) #lemma file
  print "--- Loaded lemma dict ---"
  print "dict size: ", len(lemma_dict.items())
  
  if(args.sigma_pmi):
    modelFile_SigmaPmi = open(args.out_file_prefix+".SigmaPMI",'w')
  if (args.sel_assoc):
    if(args.print_all):
      modelFile_SelStr = open(args.out_file_prefix+"SelStr",'w')
      modelFile_SelStrInverse = open(args.out_file_prefix+"SelStrInv",'w')
      modelFile_SelAssoc = open(args.out_file_prefix+"SelAssoc",'w')
      modelFile_SelAssocInverse = open(args.out_file_prefix+"SelAssocInv",'w')
      modelFile_SelAssoc_Both = open(args.out_file_prefix+"SelAssoc+Inv",'w')
    if(args.print_combine):
      modelFile_Combine = open(args.out_file_prefix+"SelAssoc_Combine",'w')



 
  print "---Computing CDF....---"
  incompletePairs, notInW2C = ComputeFreqDist(depRelFile, w2c)
  print "---Done computing CDF---"
  print "number of predicates: ", len(vbDict)
  print "number of arguments: ", len(argDict)
  print "number of tuples: ", argVbRelCFD.N()
  print "incomplete pairs: ",incompletePairs
  print "arguments not in w2c: ",notInW2C


  print "---Computing MLE PDFs....---"
  # P(arg|vb,rel)
  argVbRelPDF = ConditionalProbDist(argVbRelCFD,MLEProbDist)
  # P(arg|rel)
  argRelPDF=ConditionalProbDist(argRelCFD,MLEProbDist)
  # Inverse model
  # P(vb|arg,rel)
  vbArgRelPDF=ConditionalProbDist(vbArgRelCFD,MLEProbDist)
  # P(vb|rel)
  vbRelPDF=ConditionalProbDist(vbRelCFD,MLEProbDist)

  #print_model_file2(argVbRelPDF,argVbRelCFD,modelFile)
  #print_model_file1(vbRelPDF,vbRelCFD,modelFile)

  if(args.sigma_pmi):
    SigmaPmi = ComputeSigmaPMI(argVbRelPDF, argRelPDF, argVbRelCFD, vbDict, argDict)
    print_dict_file3(SigmaPmi,modelFile_SigmaPmi)
  if(args.sel_assoc):
    #direct
    SelStr = ComputeKLDivergence(argVbRelPDF, argRelPDF, argVbRelCFD,vbDict, argDict)
    SelAssoc = ComputeSelAssoc(argVbRelPDF, argRelPDF, argVbRelCFD, SelStr, vbDict, argDict)

    #indirect
    SelStrInverse = ComputeKLDivergence(vbArgRelPDF,vbRelPDF, vbArgRelCFD,argDict, vbDict)
    SelAssocInverse = ComputeSelAssoc(vbArgRelPDF,vbRelPDF, vbArgRelCFD,SelStrInverse, argDict, vbDict)

    if(args.print_all):
      print_dict_file2(SelStr,modelFile_SelStr)
      print_dict_file2(SelStrInverse,modelFile_SelStrInverse)
      print_dict_file3(SelAssoc,modelFile_SelAssoc)
      print_dict_file3(SelAssocInverse,modelFile_SelAssocInverse)
      print_dicts_file3(SelAssoc,SelAssocInverse,modelFile_SelAssoc_Both)

    if(args.print_combine):
      print_combine(SelAssoc, SelStr, argVbRelPDF, argRelPDF, argVbRelCFD, argRelCFD, vbRelCFD, modelFile_Combine)


if __name__ == '__main__':
  
  main()