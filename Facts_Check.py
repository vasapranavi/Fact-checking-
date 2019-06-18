# Fact checkinh by using wikipedia
# Final Version
# encoding=utf-8#
import re
import os
import nltk
import numpy as np
import pandas as pd
import csv, xlrd
from urllib import request
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from bs4.element import Tag
from nltk.corpus import wordnet as wn
from itertools import chain, groupby
from nltk.tag import StanfordNERTagger
from nltk.stem import WordNetLemmatizer

wikipedia_base = 'https://en.wikipedia.org/wiki/'
#pronoun = ['he', 'she', 'his', 'her', 'their', 'He', 'She', 'His', 'Her', 'Their']
negtive_words = ['no', 'not']

def open_target_url(subject):
    return wikipedia_base + subject
def obtain_target_text(url):
    try:
        html = request.urlopen(url).read().decode('utf-8')
        bs = BeautifulSoup(html, features = 'html.parser')
        attrs = {"class": 'mw-parser-output'}
        need = bs.find(name = 'div', attrs = attrs)
        raw_text = ''
        for tag in need.contents:
            if tag.name == 'p' :
                raw_text += tag.text.strip() + '\n'
        header = ''
        stat = []
        counter = 0
        for tag in need.contents:
            if tag.name == 'table':
                for tbody in tag.contents:
                    if tbody.name == 'tbody':
                        for tr in tbody:
                            features = ''
                            contents = ''
                            if tr.name == 'tr': 
                                for th in tr:                 
                                    if th.name == 'th':
                                        if counter == 0:
                                            header = th.text
                                            counter += 1
                                        else:
                                            features = re.sub('\n', ' ', th.text)
                            
                                    if th.name == 'td':
                                        contents = re.sub('\n', ' ', th.text)
                                stat.append(header + ' is ' + features + ' of ' + contents + '.\n')  
        raw_text += ''.join(stat)
        return raw_text
    except Exception as e:
        print('Can not find the target page in Wikipeadia.',url)
        return
def nerInNLTK(text):
    text1 = re.sub('(\.|\'s|\')', '', text)
    try:
        st = StanfordNERTagger('.//NER_//english.all.3class.distsim.crf.ser.gz', './/NER_//stanford-ner.jar')
        result = st.tag(text1.split())
        return result
    except Exception as e:
        print('did not detect any person, organization or location, please check.')
def combo_ners(word_with_tags):
    entities = []
    relations = []
    for tag, chunk in groupby(word_with_tags, lambda x: x[1]): 
        if tag != 'O':      
            entities.append((tag, " ".join(w for w, t in chunk)))
        else:
            relations.append((tag, " ".join(w for w, t in chunk)))
    #print(entities)
    #print(relations)
    entities1 = []
    for entitie in entities:
        if entitie[0] == 'PERSON':
            entities1.append( ('PERSON',re.sub('(\.|\'s|\')', '', entitie[1])))
        if entitie[0] == 'LOCATION':
            entities1.append(('LOCATION', re.sub('\.', '', entitie[1])))
            
    return (entities1, relations)
def eliminate_symbols(inpt):
    return re.sub(r'[^\w]', ' ', inpt)
def eliminate_symbols_1(inpt):
    return re.sub(r'[^\w]', '', inpt)
def eliminate_annotations(inpt):
    return re.sub(r'\[\S*\]', '', inpt)
def pre_process(sent):
    key_facts = []
    sent_token = nltk.word_tokenize(sent)
    pos_taged = nltk.pos_tag(sent_token)
    for word, tag in pos_taged:
        if tag.startswith('N') or tag.startswith('J') or tag.startswith('V') or tag.startswith('R')or tag.startswith('C'):
            key_facts.append(word)
    return key_facts
def find_similar_word(pure_key_word):
    simis_word = []
    for word in pure_key_word:
        synsets = wn.synsets(word)
        for syn in synsets:
            for sy in syn.lemmas():
                simis_word.append(sy.name())
    simis_word += pure_key_word 
    return set(simis_word)
# natural language process . para1: the sequence of the html store in the url.txt
def pre_process1(words):
    sent_token = nltk.word_tokenize(words)
    patterns= [(r'.*ing$','VBG'),(r'.*ed$','VBD'),(r'.*es$','VBZ'),(r'.*ould$','MD'),\
           (r'.*\'s$','NN$'),(r'.*ly$','RB'),(r'.*s$','NNS'),(r'^-?[0-9]+(.[0-9]+)?$','CD'),(r'.*','NN')]
    regexp_tagger = nltk.RegexpTagger(patterns)
    pos_taged = regexp_tagger.tag(sent_token)
    wnl = WordNetLemmatizer()
    after = []
    for lemma in pos_taged:
        if lemma[1].startswith('N'):
            after.append(wnl.lemmatize(lemma[0],'n'))
            continue
        if lemma[1].startswith('J'):
            after.append(wnl.lemmatize(lemma[0], 'a'))
            continue
        if lemma[1].startswith('V'):
            after.append(wnl.lemmatize(lemma[0], 'v'))
            continue
        if lemma[1].startswith('R'):
            after.append(wnl.lemmatize(lemma[0], 'r'))
            continue
        else:
            after.append(wnl.lemmatize(lemma[0]))
    return after

def key_word_match(needed_text, entities, facts, confident_threshold):
    matched_sentences = []
    sentences = sent_tokenize(needed_text)
    pre_suf = entities.split()
    pre_suf.append(entities)
    pure_key_word = [fact for fact in facts if fact not in pre_suf]
    simis_word = find_similar_word(pure_key_word)
    for sentence in sentences:
        had_matched_word = []
        words = pre_process1(sentence)
        for word in words:
            if  word in simis_word:
                    had_matched_word.append(word)
        matched_lenth = len(set(had_matched_word))
        confidence = matched_lenth - confident_threshold
        neg_matched = [neg for neg in negtive_words if neg in had_matched_word]
        neg_input = [neg for neg in negtive_words if neg in facts]
        if len(pure_key_word) <= confidence and neg_input == neg_matched:
            matched_sentences.append(sentence)
            print('Mathched ',matched_lenth, 'keywords on:', ' '.join(words))
            print('Mathched words: ', set(had_matched_word))
    return matched_sentences        

# Imports the Google Cloud client library
from google.cloud import translate
def translation(traget_lan, sentence):    
    # Instantiates a client
    translate_client = translate.Client()
    # The text to translate
    text = sentence
    # Translates some text into Chinese
    translation = translate_client.translate(
        text,
        target_language=traget_lan)
    print(u'Translation: {}'.format(translation['translatedText']))
    return translation['translatedText']
def compare_translation_sent(sent1, sent2):
    sents1 = eliminate_symbols_1(sent1)
    sents2 = eliminate_symbols_1(sent2)
    sents_array_1 = [i for i in sents1]
    sents_array_2 = [j for j in sents2]
    counter = 0
    chunk_match_length = []
    for i in range(len(sents_array_2)):
        for j in range(len(sents_array_1)):
            if i <= (len(sents_array_2) - 1):
                if sents_array_2[i] == sents_array_1[j]:
                    counter += 1
                    chunk_match_length.append(counter) 
                    i += 1
                    continue
                else:
                    counter = 0
                    continue
    if len(chunk_match_length) > 0:
        match_accuracy = sorted(chunk_match_length, reverse = True)[0] / len(sents_array_2)
    else:
        match_accuracy = 0
    return match_accuracy    
# please use pip install xlrd here.
def read_csv_files(file_name):
    df = pd.read_excel(os.getcwd() +'//'+ file_name)
    stat = ['Fact_Statement']
    target = df[stat].values.astype(str).tolist()
    target1 = np.array(target).flatten()
    return target1
stat_lines = read_csv_files('Fact_Checking.xlsx')       
 
for line in stat_lines:
    print(line)
    inputs_trans = translation('zh-CN', line)
    key_facts = pre_process(line)
    key_facts1 = pre_process1(' '.join(key_facts))
    # recognize person, location and org
    ners = nerInNLTK(line)
    entities = combo_ners(ners)[0]
    sent_mean = {}
    counter = 0;
    for entitie in entities:
        target_url = open_target_url(entitie[1])
        target_text = obtain_target_text(target_url)
        if target_text == None:
            break
        target_text_1 = eliminate_annotations(target_text)
        matched_sentences = key_word_match(target_text_1, entitie[1], key_facts1, 0)
        if len(matched_sentences) > 0:
            for sent in range(len(matched_sentences)):
                counter += 1
                print('Matched ', counter, ': ', matched_sentences[sent])
        else:
            print("No Matches", 'in Wikipedia page of', '\"', entitie[1],'.\"')
         
    print()


for sent in matched_sentences:
    matched_trans = translation('zh-CN', sent)
    mean_similarity = compare_translation_sent(matched_trans, inputs_trans)
    sent_mean[sent] = mean_similarity
    max_mean = sorted(sent_mean.items(), key=lambda item: item[1], reverse=True)[0]
    if float(max_mean[1]) > 0.21 and counter == 0:
       counter += 1
       print('Matched the sentence with max semantic similarity of :', max_mean, end='\n\n')
    else:
       print('You maybe worng!', end='\n\n')
