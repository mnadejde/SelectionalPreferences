import re
import math

#### MANAGE DICT OBJECTS ####

def increment_dict(dict,arg,value):
  if dict.has_key(arg):
    dict[arg]+=value
  else:
    dict[arg]=value

def print_dict(dict):
	keys = dict.keys()
	keys.sort()
	for key in keys:
		print key.encode('utf-8')+"\t"+str(dict[key])

def print_dict_file2(dict,file):
	keys = dict.keys()
	keys.sort()
	for key in keys:
		print >> file, '%s\t%s\t%0.7f'%(key[0], key[1],dict[key])

def print_dict_file3(dict,file):
	keys = dict.keys()
	keys.sort()
	for key in keys:
		print >> file, '%s\t%s\t%s\t%0.7f'%(key[0], key[1], key[2],dict[key])

def print_dicts_file3(dict1,dict2,file):
	keys = dict1.keys()
	keys.sort()
	SelAssoc_sorted = sorted(dict1.items(),  key=lambda k: (k[0][0], k[0][1], k[1], k[0][2]))
	for key, value in SelAssoc_sorted:
		if dict2.has_key((key[2],key[1],key[0])):
			print >> file, '%s\t%s\t%s\t%0.7f\t%0.7f'%(key[1], key[0], key[2], value,dict2[(key[2],key[1],key[0])])
		else:
			print >> file, '%s\t%s\t%s\t%0.7f\t%0.7f'%(key[1], key[0], key[2], value, 0.0)

def print_dict_file(dict,file):
	keys = dict.keys()
	keys.sort()
	for key in keys:
		print >> file, '%s\t%0.7f'%(key.encode('utf-8'), dict[key])

def print_dict_file2s(dict,file):
	keys = dict.keys()
	keys.sort()
	for key in keys:
		print >> file, '%s\t%s'%(key, str(dict[key]))

def print_combine_file2s(dict1, dict2,file):
	keys = dict1.keys()
	keys.sort()
	for key in keys:
		if dict2.has_key(key):
			print >> file, '%s\t%s\t%f0.2'%(key, str(dict1[key]),dict2[key])
		else:
			print >>file, '%s\t%s\t%s'%(key, str(dict1[key]),"-1")

def print_model_file2(PDF,CFD,file):
  for condition in CFD.conditions():
    for sample in CFD[condition].items():
      print >> file, '%s\t%s\t%s\t%0.7f'%(condition[0],condition[1],sample[0],PDF[condition].prob(sample[0]))

def print_model_file1(PDF,CFD,file):
  for condition in CFD.conditions():
    for sample in CFD[condition].items():
      print >> file, '%s\t%s\t%0.7f'%(condition,sample[0],PDF[condition].prob(sample[0]))

def print_combine(SelAssoc, SelStr, argVbRelPDF, argRelPDF, argVbRelCFD, argRelCFD, vbRelCFD, file):
  # key = (verb, rel, arg)
  # sort by verb, rel, SelAssoc score
  SelAssoc_sorted = sorted(SelAssoc.items(),  key=lambda k: (k[0][0], k[0][1], k[1], k[0][2]))
  for key,value in SelAssoc_sorted:
    vb, rel, arg = key
    sel_assoc = value
    sel_str = SelStr[(vb, rel)]
    p_argVbRel = math.log(argVbRelPDF[(vb,rel)].prob(arg))
    p_argRel = math.log(argRelPDF[rel].prob(arg))
    c_argVbRel = argVbRelCFD[(vb,rel)][arg]
    c_argRel = argRelCFD[rel][arg]
    c_vbRel = vbRelCFD[rel][vb]
    # vb, rel, arg, sel_assoc, sel_str, p_argVbRel, p_argRel, c_argVbRel, c_argRel, c_vbRel
    print >> file, '%s\t%s\t%s\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t%d\t%d\t%d'%(vb, rel, arg, sel_assoc, sel_str, p_argVbRel, p_argRel, c_argVbRel, c_argRel, c_vbRel)

def sord_dict(key1,key2):
  vb,arg = key.split("\t")
  return

def ReadDictFromFile(fileName,dict):
  f = open(fileName,'r')
  line = f.readline().split()
  while line and len(line)>2:
    increment_dict(dict,str(line[0]).strip().encode('utf-8')+"\t"+str(line[1]).strip().encode('utf-8'),float(line[2]))
    #prob_table[line[0]+"\t"+line[1]]=float(line[2])
    line = f.readline().split()

def ReadDictFromFile2(fileName,dict):
  f = open(fileName,'r')
  line = f.readline().split()
  while line and len(line)>1:
    #I have problems with encoding when I create the lemma file
    dict[str(line[0]).strip().lower()]=str(line[1]).strip().lower()
    line = f.readline().split()

##### FILTER FOR XML/SYTNAX STRINGS #####

def filter_in(line_xml):
  line_xml=re.sub(r"&#124;","BAR",line_xml)
  line_xml=re.sub(r"&bar;","BAR",line_xml)
  line_xml=re.sub(r"&#91;","[",line_xml)
  line_xml=re.sub(r"&#93;","]",line_xml)

def filter_out(out_xml):
  out_xml= re.sub(r"&amp;bar;","&#124;",out_xml)
  out_xml= re.sub(r"&amp;#124;","&#124;",out_xml)
  out_xml= re.sub(r"&amp;apos;","&apos;",out_xml)
  out_xml= re.sub(r"&amp;quot;","&quot;",out_xml)
  out_xml= re.sub(r"&amp;gt;","&gt;",out_xml)
  out_xml= re.sub(r"&amp;lt;","&lt;",out_xml)

##### FILTERS FOR VERBS AND ARGUMENTS #####

Pronouns = ['i','he','she','we','you','they','it','me','them']

def lemmatize(arg):
  if lemma_dict.has_key(arg.lower()):
    #print "has lemma: ",arg, lemma_dict[arg]
    return lemma_dict[arg.lower()]
  else:
    return arg

def filter_vb(vb):
  # alpha filter should eliminate all punctuation, numbers, non-ascii characters
  non_alpha = re.compile('[^a-zA-Z]+')
  # this allows for split auxiliaries 're 's 've
  aux = re.compile('^\'(re|ll|ve|d|m|s)$')
  if not non_alpha.search(vb) or aux.match(vb):
    return 1
  else:
    return 0

def filter_arg(arg):
  #check if it's non-alphanumeric
  non_alpha = re.compile('[\W]+')
  web = re.compile(r'\bhttp|\bwww')
  date = re.compile('([0-9]+[\.|\-|\/]?)+')
  nr = re.compile('[0-9]+')
  
  #check if only digit
  #check if it starts with www
  if non_alpha.search(arg) :
    if web.match(arg):
      return "WWW"
    if date.match(arg):
      return "DDAATTEE"
    return "NON_ALPHA"
  else:
    if arg.isdigit():
      return "NNRR"
    if arg.lower() in Pronouns:
      return "PRN"
    else:
      return arg

##### READ IN THE COLLINS HEAD RULES FILE #####

def read_head_rules(f,rule_table):
	rule=f.readline().split() #nr_NT parent_NT orientation_0(head_final->reverse)_1(head_initial) possible_head_children_NTs
	while rule:
		rule_table[rule[1]]=(rule[2],rule[3:])#(rule[2],rule[3,])
		rule=f.readline().split()
	return rule_table