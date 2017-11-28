#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from gensim.summarization.syntactic_unit import SyntacticUnit
from gensim.parsing.preprocessing import preprocess_documents
from gensim.utils import tokenize
from six.moves import xrange
import re
import logging
from hazm import *

logger = logging.getLogger('summa.preprocessing.cleaner')

try:
    #from pattern.en import tag
    from hazm import POSTagger
    tagger = POSTagger(model='resources/postagger.model')
    logger.info("'pattern' package found; tag filters are available for Persian")
    HAS_PATTERN = True
except ImportError:
    #logger.info("'pattern' package not found; tag filters are not available for English")
    logger.info("'pattern' package not found; tag filters are not available for Persian")
    HAS_PATTERN = False


SEPARATOR = r'@'
RE_SENTENCE = re.compile(r'(\S.+?[.!?])(?=\s+|$)|(\S.+?)(?=[\n]|$)', re.UNICODE)  # backup (\S.+?[.!?])(?=\s+|$)|(\S.+?)(?=[\n]|$)
AB_SENIOR = re.compile(r'([A-Z][a-z]{1,2}\.)\s(\w)', re.UNICODE)
AB_ACRONYM = re.compile(r'(\.[a-zA-Z]\.)\s(\w)', re.UNICODE)
AB_ACRONYM_LETTERS = re.compile(r'([a-zA-Z])\.([a-zA-Z])\.', re.UNICODE)
UNDO_AB_SENIOR = re.compile(r'([A-Z][a-z]{1,2}\.)' + SEPARATOR + r'(\w)', re.UNICODE)
UNDO_AB_ACRONYM = re.compile(r'(\.[a-zA-Z]\.)' + SEPARATOR + r'(\w)', re.UNICODE)


#def split_sentences(text):
    #processed = replace_abbreviations(text)
    #return [undo_replacement(sentence) for sentence in get_sentences(processed)]

def split_sentences(text):
    return (sent_tokenize(text))

def replace_abbreviations(text):
    return replace_with_separator(text, SEPARATOR, [AB_SENIOR, AB_ACRONYM])


def undo_replacement(sentence):
    return replace_with_separator(sentence, r" ", [UNDO_AB_SENIOR, UNDO_AB_ACRONYM])


def replace_with_separator(text, separator, regexs):
    replacement = r"\1" + separator + r"\2"
    result = text
    for regex in regexs:
        result = regex.sub(replacement, result)
    return result


#def get_sentences(text):
#    for match in RE_SENTENCE.finditer(text):
#        yield match.group()

def get_sentences(text):
    te = sent_tokenize(text)
    for each in te:
        yield (each)


def merge_syntactic_units(original_units, filtered_units, tags=None):
    units = []
    for i in xrange(len(original_units)):
        if filtered_units[i] == '':
            continue

        text = original_units[i]
        token = filtered_units[i]

        if tags :
            try:
                tag = tags[i][1]
            except:
                tag = None
        else:
            tag = None

        #tag = tags[i][1] if tags else None
        
        sentence = SyntacticUnit(text, token, tag)
        sentence.index = i

        units.append(sentence)

    return units


def join_words(words, separator=" "):
    return separator.join(words)


def clean_text_by_sentences(text):
    """ Tokenizes a given text into sentences, applying filters and lemmatizing them.
    Returns a SyntacticUnit list. """
    original_sentences = split_sentences(text)
    filtered_sentences = [join_words(sentence) for sentence in preprocess_documents(original_sentences)]
    tags = clean_text_by_word(text)
    return merge_syntactic_units(original_sentences, filtered_sentences, tags)


def clean_text_by_word(text, deacc=True):
    """ Tokenizes a given text into words, applying filters and lemmatizing them.
    Returns a dict of word -> syntacticUnit. """
    text_without_acronyms = replace_with_separator(text, "", [AB_ACRONYM_LETTERS])
    original_words = list(tokenize(text_without_acronyms, to_lower=True, deacc=deacc))
    filtered_words = [join_words(word_list, "") for word_list in preprocess_documents(original_words)]
    if HAS_PATTERN:
        tags = tagger.tag(original_words) # tag needs the context of the words in the text
    else:
        tags = None
    units = merge_syntactic_units(original_words, filtered_words, tags)
    return dict((unit.text,unit) for unit in units)


def tokenize_by_word(text):
    text_without_acronyms = replace_with_separator(text, "", [AB_ACRONYM_LETTERS])
    return tokenize(text_without_acronyms, to_lower=True, deacc=True)
