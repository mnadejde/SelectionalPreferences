import fileinput
import sys
import re
import itertools
import math
import numpy as np
import operator
import cStringIO

from Util import *

import nltk
# module for computing a Conditional Frequency Distribution
from nltk.probability import ConditionalFreqDist

# module for computing a Conditional Probability Distribution
from nltk.probability import ConditionalProbDist

# module for computing a probability distribution with the Maximum Likelihood Estimate
from nltk.probability import MLEProbDist

# module for computing a Frequency Distribution
from nltk.probability import FreqDist

##################
## Estimate P(arg|vb,rel)
## Input: file with tuples (vb,rel,arg), one on each line, extracted using StanfordDependencies from the parsed side of the parlalle corpus
## arg and vb are words and rel can be one of nsubj,nsubjpas, pobj etc.
## Computes a conditional prob distr with written bell smoothing
## Output: models for P_WB(arg|vb,rel), P_WB(arg|vb), P(arg), lambda1, lambda2

#we need 2 Conditional Frequency Distributions
# F(arg|vb,rel)
argVbRelCFD=ConditionalFreqDist()
# F(arg|vb) for backoff
argVbCFD=ConditionalFreqDist()
# F(arg) for backoff
argFD=FreqDist()

# F(rel) -> used to add the rel to the vocabulary
relFD=FreqDist()

# P_MLE(arg|vb,rel)
#argVbRelPDF
# P_MLE(arg|vb) for backoff
#argVbPDF
# P_MLE(arg) for backoff
#argPDF

# lambda1, lambda2 for interpolating the backoff distribution with the MLE distribution
lambda1 = 0.0
lambda2 = 0.0

lemma_dict = {}

#following = FreqDist()


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

def ComputeFreqDist(f):
  incompletePairs=0
  line = f.readline()
  while(line):
    tokens = line.split()
    if(len(tokens)>2):
      rel=tokens[0]
      vb=tokens[1]
      arg=tokens[2]
    
      if filter_vb(vb):
        argVbRelCFD[(lemmatize(rel),lemmatize(vb))][filter_arg(lemmatize(arg))]+=1
        argVbCFD[lemmatize(vb)][filter_arg(lemmatize(arg))]+=1
        argFD[filter_arg(lemmatize(arg))]+=1
        relFD[rel]+=1
    #following[(lemmatize(rel),lemmatize(vb),filter_arg(lemmatize(arg)))]+=1
    else:
      incompletePairs+=1
    line = f.readline()
  return incompletePairs

def ComputeWBArg(argPDF):
  lambda3 = 1.0*argFD.N()/(argFD.N()+argFD.B())
  lambda3_c = 1-lambda3
  uniform = 1.0/argFD.B() # could be the size of a Vocabulary of seen and unseen words -> here is the number of seen arguments ???
  PWB = dict()
  for arg in argFD:
    PWB[arg]=lambda3 * argPDF.prob(arg) + lambda3_c * uniform
  return PWB, lambda3_c * uniform


def ComputeWBVbArg(argVbPDF,argPDFWB):
  #lambda2 =  1.0 * argVbCFD.N()/ (len(argVbCFD.conditions())+argVbCFD.N())
  #lambda2_c = 1-lambda2
  count = 0
  T=0
  PWB = dict()
  PWB_backoff_weight = dict()
#  for condition in argVbCFD.conditions():
#    T+=argVbCFD[condition].B()
#  lambda2 =  1.0 * argVbCFD.N()/ (T+argVbCFD.N())
#  lambda2_c = 1-lambda2
  
  # condition is vb
  for condition in argVbCFD.conditions():
    PWB[condition]= dict()
    T=argVbCFD[condition].B()
    lambda2 =  1.0 * argVbCFD[condition].N()/ (T+argVbCFD[condition].N())
    lambda2_c = 1-lambda2
    # sample is (arg, prob)
    for sample in argVbCFD[condition].items():
      PWB[condition][sample[0]] = lambda2 * argVbPDF[condition].prob(sample[0]) + lambda2_c * argPDFWB[sample[0]] #should be argPDF_WB[arg] ? but what does that meand
      PWB_backoff_weight[condition]=lambda2_c
      count+=1
  #PWB[condition].pop(0)
  return PWB,PWB_backoff_weight, count

def ComputeWBRelVbArg(argVbRelPDF,argVbPDFWB):
#  lambda1 =  1.0 * argVbRelCFD.N()/ (len(argVbRelCFD.conditions())+argVbRelCFD.N())
#  lambda1_c = 1-lambda1
  count = 0
  PWB = dict()
  PWB_backoff_weight = dict()
  #condition is (rel,vb)
  for condition in argVbRelCFD.conditions():
    PWB[condition]= dict()
    T=argVbRelCFD[condition].B()
    lambda1 =  1.0 * argVbRelCFD[condition].N()/ (T+argVbRelCFD[condition].N())
    lambda1_c = 1-lambda1
    #sample is (arg,prob)
    for sample in argVbRelCFD[condition].items():
      PWB[condition][sample[0]]= lambda1 * argVbRelPDF[condition].prob(sample[0]) + lambda1_c * argVbPDFWB[condition[1]][sample[0]] #should be argPDF_WB[arg] ? but what does that meand
      PWB_backoff_weight[condition]= lambda1_c
      count+=1
  return PWB,PWB_backoff_weight, count

def WriteToArpaFormat(modelFile, n1,n2,n3,argPDFWB,argVbPDFWB,argRelVbPDFWB, uniform, argVbPDFWB_backoff_weights, argRelVbPDFWB_backoff_weights):
  print "---Saving models to file---"


  
  # ARPA format ?
  # PWB(z|xy) x y z BOW(x y)
  # PWB(arg|rel vb) rel vb arg BOW(rel, vb)

  #should be log10(prob) ?? or log2 ->check moses !! probably ln
  temp = cStringIO.StringIO()
  for sample in sorted(argPDFWB.items(),key=operator.itemgetter(0)):
    if argVbPDFWB_backoff_weights.has_key(sample[0]):
      #the case where the arg can also be a verb and we shouldn't list is twice as 1-gram -> we don't need this for 2-grams because rel vb will never appear instead of vb arg because of the rel
      #so we print lambda(verb) here and eliminated from the backoff_vb dictionary
      print >> temp, '%0.7f\t%s\t%0.7f'%(math.log10(sample[1]), sample[0], math.log(argVbPDFWB_backoff_weights[sample[0]]))
      del argVbPDFWB_backoff_weights[sample[0]]
    else:
      print >> temp, '%0.7f\t%s'%(math.log10(sample[1]), sample[0]) #backoff weight -> only needed if the n-gram is a prefix for a higher n-gram

  #first we computed how many 1-grams we have without duplicates (args that are aslo vb) and now we print the numbers
  print >> modelFile, '\\data\\'
  print >> modelFile, 'ngram 1=%d'%(n1+len(relFD)+len(argVbPDFWB_backoff_weights)+1) #+ number of relations + number of vb conditions in backoff_weights + <unk>
  print >> modelFile, 'ngram 2=%d'%(n2+len(argRelVbPDFWB_backoff_weights)) #+ number of rel vb conditions in backoff_weights
  print >> modelFile, 'ngram 3=%d'%(n3)
  print >> modelFile, ''

  #now print 1-grams without duplicates
  #The StringIO object can accept either Unicode or 8-bit strings, but mixing the two may take some care. If both are used, 8-bit strings that cannot be interpreted as 7-bit ASCII (that use the 8th bit) will cause a UnicodeError to be raised when getvalue() is called.
  print >> modelFile, '\\1-grams:'
  for rel in relFD.items():
    print >> modelFile, '%0.1f\t%s'%(-99,rel[0])
  print >> modelFile, temp.getvalue(),
  temp.close()
  print >> modelFile, '%0.7f\t%s'%(math.log10(uniform), "<unk>")

  #print the backoff weights lambda(vb) which appear as unigrams in ARPA file with inf probability (-99) and the backoff weights
  for sample in sorted(argVbPDFWB_backoff_weights.items(),key=operator.itemgetter(0)):
    print >> modelFile, '%0.1f\t%s\t%0.7f'%(-99,sample[0], math.log10(sample[1])) #backoff weight -> only needed if the n-gram is a prefix for a higher n-gram

  print >> modelFile, ''
  
  print >> modelFile, '\\2-grams:'
  for sample in sorted(argVbPDFWB.items(),key=operator.itemgetter(0)):
    for arg in sorted(sample[1].items(),key=operator.itemgetter(1), reverse=True):
      # condition=(rel,vb) arg PWB backoff_weight_for condition
      #PWB(arg|vb) vb arg BOW(vb)
      print >> modelFile, '%0.7f\t%s\t%s'%( math.log10(arg[1]),sample[0],arg[0])

  #print the backoff weights lambda(rel vb)
  for sample in sorted(argRelVbPDFWB_backoff_weights.items(),key=operator.itemgetter(0)):
    print >> modelFile, '%0.1f\t%s\t%s\t%0.7f'%(-99,sample[0][0],sample[0][1],math.log10(sample[1]))
  print >> modelFile, ''
  
  print >> modelFile, '\\3-grams:'
  for sample in sorted(argRelVbPDFWB.items(),key=operator.itemgetter(0)):
    for arg in sorted(sample[1].items(),key=operator.itemgetter(1), reverse=True):
      # PWB(arg|rel vb) rel vb arg BOW(rel, vb)
      print >> modelFile, '%0.7f\t%s\t%s\t%s'%(math.log10(arg[1]),sample[0][0],sample[0][1],arg[0])
  print >> modelFile, ''
  print >> modelFile, '\\end\\'
  
  return
  
  # format
  # nr of 1-gram, 2-gram ,3-gram
  # lambdaC_1 lambdaC_2 lambdaC_3 -> in case
  # PWB for each model (NOT already multiplied with the back-off weight -> lambdac_1 * P(arg|vb))
  # -> actually we know that the lambda will only depend on the condition vb so we don't need the information about vb at runtime
  # -> if we can find Pwb(arg|vb) then lambda2 depends on the vb from Pwb(arg|vb) which was seen in training -> but DARPA format lM requires Pwb(arg|vb) lambda2(vb)
  # for lm (vb,arg) could be an n-gram on it's own but in our case it will always be the back-off
  # I could try creating the darpa format and using kenlm because (vb,arg) will never be queried -> only reached while backingof
  print >> modelFile, '%d\t%d\t%d'%(len(argPDFWB),countArgVB,countRelVbArg)
  #lambda3_c*uniform (backoff for unseen args)
  print >> modelFile, '%0.7f'%(backoff_uniform)
  #print_dict_file(argPDFWB,modelFile)
  for sample in sorted(argPDFWB.items(),key=operator.itemgetter(0)):
    print >> modelFile, '%s\t%0.7f\t%0.7f'%(sample[0], sample[1], argPDF.prob(sample[0]))
  
  # comparing the WB prob with the MLE prob -> if the arg is very probable then PWB(vb,arg) will be higher than for pairs with arg not so common -> if the arg PRN the WB is rewarded a lot ??? it's heigher than the MLE prob??
  for sample in sorted(argVbPDFWB.items(),key=operator.itemgetter(0)):
    for arg in sorted(sample[1].items(),key=operator.itemgetter(1), reverse=True):
      # condition=(rel,vb) arg PWB backoff_weight_for condition
      print >> modelFile, '%s\t%s\t%0.7f\t%0.7f\t%0.7f'%(sample[0],arg[0], arg[1],argVbPDF[sample[0]].prob(arg[0]) ,argVbPDFWB_backoff_weights[sample[0]])
  
  for sample in sorted(argRelVbPDFWB.items(),key=operator.itemgetter(0)):
    for arg in sorted(sample[1].items(),key=operator.itemgetter(1), reverse=True):
      print >> modelFile, '%s\t%s\t%s\t%0.7f\t%0.7f\t%0.7f'%(sample[0][0],sample[0][1],arg[0], arg[1], argVbRelPDF[sample[0]].prob(arg[0]), argRelVbPDFWB_backoff_weights[sample[0]])
  #print sample[0],sample[1]

def OrigPrint():
  print "---Saving models to file---"
  # format
  # nr of 1-gram, 2-gram ,3-gram
  # lambdaC_1 lambdaC_2 lambdaC_3 -> in case
  # PWB for each model (NOT already multiplied with the back-off weight -> lambdac_1 * P(arg|vb))
  # -> actually we know that the lambda will only depend on the condition vb so we don't need the information about vb at runtime
  # -> if we can find Pwb(arg|vb) then lambda2 depends on the vb from Pwb(arg|vb) which was seen in training -> but DARPA format lM requires Pwb(arg|vb) lambda2(vb)
  # for lm (vb,arg) could be an n-gram on it's own but in our case it will always be the back-off
  # I could try creating the darpa format and using kenlm because (vb,arg) will never be queried -> only reached while backingof
  print >> modelFile, '%d\t%d\t%d'%(len(argPDFWB),countArgVB,countRelVbArg)
  #lambda3_c*uniform (backoff for unseen args)
  print >> modelFile, '%0.7f'%(backoff_uniform)
  #print_dict_file(argPDFWB,modelFile)
  for sample in sorted(argPDFWB.items(),key=operator.itemgetter(0)):
    print >> modelFile, '%s\t%0.7f\t%0.7f'%(sample[0], sample[1], argPDF.prob(sample[0]))
  
  # comparing the WB prob with the MLE prob -> if the arg is very probable then PWB(vb,arg) will be higher than for pairs with arg not so common -> if the arg PRN the WB is rewarded a lot ??? it's heigher than the MLE prob??
  for sample in sorted(argVbPDFWB.items(),key=operator.itemgetter(0)):
    for arg in sorted(sample[1].items(),key=operator.itemgetter(1), reverse=True):
      # condition=(rel,vb) arg PWB backoff_weight_for condition
      print >> modelFile, '%s\t%s\t%0.7f\t%0.7f\t%0.7f'%(sample[0],arg[0], arg[1],argVbPDF[sample[0]].prob(arg[0]) ,argVbPDFWB_backoff_weights[sample[0]])
  
  for sample in sorted(argRelVbPDFWB.items(),key=operator.itemgetter(0)):
    for arg in sorted(sample[1].items(),key=operator.itemgetter(1), reverse=True):
      print >> modelFile, '%s\t%s\t%s\t%0.7f\t%0.7f\t%0.7f'%(sample[0][0],sample[0][1],arg[0], arg[1], argVbRelPDF[sample[0]].prob(arg[0]), argRelVbPDFWB_backoff_weights[sample[0]])
  #print sample[0],sample[1]

def main():
  DEBUG =1
  depRelFile=open(sys.argv[1],'r')	#file with dep rel tuples
  ReadDictFromFile(sys.argv[2],lemma_dict) #lemma file
  modelFile = open(sys.argv[3],'w')
  if (len(sys.argv)==5):
    DEBUG = int(sys.argv[4])

  print "---Done loading lemma file---"
 
  print "---Computing CDF....---"
  incompletePairs = ComputeFreqDist(depRelFile)
  print "---Done computing CDF---"
  print "incomplete pairs: ",incompletePairs
  
  if(DEBUG):
  
    print "Info about F(arg)"
    print "unique samples: ", argFD.B()
    print "total seen samples: ", argFD.N()
    print "top arg:", argFD.max()
    print "count for support: ", argFD['support']
    print "Info about CFD(arg|rel,vb)"
    print "unique conditions seen: ", len(argVbRelCFD.conditions())
    print "total seen samples", argVbRelCFD.N()
    top_CFD1 = sorted(argVbRelCFD[('dobj','enjoy')].items(),key=operator.itemgetter(1), reverse=True)[:10]
    print "all dobj,enjoy: ", argVbRelCFD[('dobj','enjoy')].N()
    print "top dobj for enjoy:\n",top_CFD1
    print "Info about CFD(arg|vb)"
    print "unique conditions seen: ", len(argVbCFD.conditions())
    print "total seen samples", argVbCFD.N()
    top_CFD2 = sorted(argVbCFD['enjoy'].items(),key=operator.itemgetter(1), reverse=True)[:10]
    print "all enjoy: ", argVbCFD['enjoy'].N()
    print "top arg for enjoy:\n",top_CFD2


  print "---Computing MLE PDFs....---"
  argVbRelPDF = ConditionalProbDist(argVbRelCFD,MLEProbDist)
  argVbPDF = ConditionalProbDist(argVbCFD,MLEProbDist)
  argPDF = MLEProbDist(argFD)


    #I'm not sure here Types is equivelent with argVbRelCFD.conditions() or unique condition+arg
    #!!!!! lambda should be for each history P(a|v) T = count of unique (v a) pairs starting with v
    # for each condition v -> sum(CFD[v].B() -> how many unique arguments I've seen after this condition)
   

  print "---Computing Witten-Bell smoothed PDFs....---"

  #for unseen pairs we multiply the backoff_weight with the probability of the backoff model
  #e.g. if  c(rel,vb,arg)=0 and c(vb,arg)>0 then P(arg|rel,vb)=argRelVbPDFWB_backoff_weights[(rel,vb)] * argVbPDFWB[vb].prob(arg)
  argPDFWB, backoff_uniform = ComputeWBArg(argPDF)
  argVbPDFWB, argVbPDFWB_backoff_weights,  countArgVB = ComputeWBVbArg(argVbPDF,argPDFWB)
  argRelVbPDFWB,argRelVbPDFWB_backoff_weights, countRelVbArg = ComputeWBRelVbArg(argVbRelPDF,argVbPDFWB)


  if(DEBUG):
    print "P(support|dobs,enjoy)"
    print argVbRelPDF[('dobj','enjoy')].prob('support')
    print argRelVbPDFWB[('dobj','enjoy')]['support']
    print "No args following (dobj,enjoy)", argVbRelCFD[('dobj','enjoy')].B()
    print "P(support|enjoy)"
    print argVbPDF['enjoy'].prob('support')
    print argVbPDFWB['enjoy']['support']
    print "P(support)"
    print argPDF.prob('support')
    print argPDFWB['support']

  WriteToArpaFormat(modelFile, len(argPDFWB),countArgVB,countRelVbArg,argPDFWB,argVbPDFWB,argRelVbPDFWB, backoff_uniform, argVbPDFWB_backoff_weights, argRelVbPDFWB_backoff_weights)

  if(DEBUG):
    #print sorted(argVbPDFWB['enjoy'],key=operator.itemgetter(1), reverse=True)[:5] #[('enjoy','support')]
    
    for condition in argVbPDFWB.keys()[:10]:
      sum1 = 0
      sum2 = 0
      for prob in argVbPDFWB[condition].values():
        sum1+=prob
      for arg in argVbCFD[condition].items():
        sum2+=argVbPDF[condition].prob(arg[0])
      print "total prob: ", sum1, sum2

    print "P_WB(support|dobj, enjoy)"
    print argRelVbPDFWB[('dobj','enjoy')]['support']

    for condition in argRelVbPDFWB.keys()[:10]:
      sum = 0
      for prob in argRelVbPDFWB[condition].values():
        sum+=prob
      print "total prob: ", sum

##### WB -> diversity of predicted words (args) for a history (rel,vb)
##### KN -> diversity of history (rel,vb) for a predicted word (args)
#### am I interested in both? diversity of (rel,vb, *) and of (*,arg)
#### could I have lambda's considering both?
### does KN actually include WB?

# The more 'promiscuous' a word, the more types observed that follow it, the more probability mass we reserve for unseen words that might follow this word, the more we lower its other MLE estimates

if __name__ == '__main__':
	
	main()