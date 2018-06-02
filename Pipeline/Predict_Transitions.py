import datetime
import math
import numpy as np
import pandas as pd
import pickle
import re
import sys

import textedit

from bs4 import BeautifulSoup

# Define text formatting and bill replacement logic.  
# This converts all utterances to entirely lowercase, 
# and replaces the following instances of words in an 
# utterance with the tag BILL_ID.

bill_id_pattern_1_1 = "ab[0-9]+"
bill_id_pattern_1_2 = "sb[0-9]+"
bill_id_pattern_1_3 = "aca[0-9]+"
bill_id_pattern_1_4 = "acr[0-9]+"
bill_id_pattern_1_5 = "ajr[0-9]+"
bill_id_pattern_1_6 = "ar[0-9]+"
bill_id_pattern_1_7 = "hr[0-9]+"
bill_id_pattern_1_8 = "sca[0-9]+"
bill_id_pattern_1_9 = "scr[0-9]+"
bill_id_pattern_1_10 = "sjr[0-9]+"

bill_id_pattern_2_1 = ["ab", "[0-9]+"]
bill_id_pattern_2_2 = ["sb", "[0-9]+"]
bill_id_pattern_2_3 = ["aca", "[0-9]+"]
bill_id_pattern_2_4 = ["acr", "[0-9]+"]
bill_id_pattern_2_5 = ["ajr", "[0-9]+"]
bill_id_pattern_2_6 = ["ar", "[0-9]+"]
bill_id_pattern_2_7 = ["hr", "[0-9]+"]
bill_id_pattern_2_8 = ["sca", "[0-9]+"]
bill_id_pattern_2_9 = ["scr", "[0-9]+"]
bill_id_pattern_2_10 = ["sjr", "[0-9]+"]

bill_id_pattern_3_1 = ["assembly", "bill", "[0-9]+"]
bill_id_pattern_3_2 = ["senate", "bill", "[0-9]+"]

bill_id_pattern_4_1 = ["assembly", "bill", "number", "[0-9]+"]
bill_id_pattern_4_2 = ["senate", "bill", "number", "[0-9]+"]

# model parameters
default_model_type = "NB"
default_model_filename = "nb_model.p"
default_count_vectorizer_filename = "nb_count_vect.p"
default_cleaned_raw_filename = "cleaned_transcript.csv2"

model_type = default_model_type
model_filename = default_model_filename
count_vectorizer_filename = default_count_vectorizer_filename
cleaned_raw_filename = default_cleaned_raw_filename

THRESHOLD_PROBABILITY = .5

def re_match_lists_helper(pattern_list, word_list):
    for p in range(len(pattern_list)):
        if not (re.match(pattern_list[p], word_list[p])):
            return False
    return True

def re_match_lists(pattern_list_list, word_list):
    for pl in range(len(pattern_list_list)):
        if (re_match_lists_helper(pattern_list_list[pl], word_list)):
            return True
    return False

def matches_any_4_word_pattern(word1, word2, word3, word4):
    pattern_list_list = [bill_id_pattern_4_1, bill_id_pattern_4_2]
    word_list = [word1, word2, word3, word4]
    
    return re_match_lists(pattern_list_list, word_list)

def matches_any_3_word_pattern(word1, word2, word3):
    pattern_list_list = [bill_id_pattern_3_1, bill_id_pattern_3_2]
    word_list = [word1, word2, word3]
    
    return re_match_lists(pattern_list_list, word_list)
    
def matches_any_2_word_pattern(word1, word2):
    pattern_list_list = [bill_id_pattern_2_1, bill_id_pattern_2_2,
                         bill_id_pattern_2_3, bill_id_pattern_2_4,
                         bill_id_pattern_2_5, bill_id_pattern_2_6,
                         bill_id_pattern_2_7, bill_id_pattern_2_8,
                         bill_id_pattern_2_9, bill_id_pattern_2_10]
    word_list = [word1, word2]
    
    return re_match_lists(pattern_list_list, word_list)

def matches_any_1_word_pattern(word):
    return (re.match(bill_id_pattern_1_1, word) or
            re.match(bill_id_pattern_1_2, word) or
            re.match(bill_id_pattern_1_3, word) or
            re.match(bill_id_pattern_1_4, word) or
            re.match(bill_id_pattern_1_5, word) or
            re.match(bill_id_pattern_1_6, word) or
            re.match(bill_id_pattern_1_7, word) or
            re.match(bill_id_pattern_1_8, word) or
            re.match(bill_id_pattern_1_9, word) or
            re.match(bill_id_pattern_1_10, word))

def shift_words_over(words, word_ix, shift_amount):
    words_length = len(words)
    
    for i in range(word_ix, words_length - shift_amount):
        words[i] = words[i+shift_amount]
    while(len(words) > (words_length-shift_amount)):
        del words[-1]
        
    return words

def replace_bill_ids_in_utterance(utterance, last_bill_number, t1, t2, t3, t4):
    words = utterance.lower().split()
    utterance_length = len(words)
    word_ix = 0
    bill_id_replaced = False
    while(word_ix < utterance_length):
        if (word_ix < (utterance_length-3) and
            matches_any_4_word_pattern(words[word_ix],
                                         words[word_ix+1],
                                         words[word_ix+2],
                                         words[word_ix+3])):
            last_bill_number = words[word_ix+3]
            words[word_ix] = "<BILL_ID>"
            words = shift_words_over(words, word_ix+1, 3)
            utterance_length -= 3
            bill_id_replaced = True
            t4 += 1
        elif (word_ix < (utterance_length-2) and
              matches_any_3_word_pattern(words[word_ix],
                                         words[word_ix+1],
                                         words[word_ix+2])):
            last_bill_number = words[word_ix+2]
            words[word_ix] = "<BILL_ID>"
            words = shift_words_over(words, word_ix+1, 2)
            utterance_length -= 2
            bill_id_replaced = True
            t3 += 1
        elif (word_ix < (utterance_length-1) and
            matches_any_2_word_pattern(words[word_ix],
                                         words[word_ix+1])):
            last_bill_number = words[word_ix+1]
            words[word_ix] = "<BILL_ID>"
            words = shift_words_over(words, word_ix+1, 1)
            utterance_length -= 1
            bill_id_replaced = True
            t2 += 1
        elif (matches_any_1_word_pattern(words[word_ix])):
            last_bill_number = words[word_ix].split("[a-z]+")[-1]
            words[word_ix] = "<BILL_ID>"
            bill_id_replaced = True
            t1 += 1

        word_ix += 1
            
    return (" ".join(words), last_bill_number, bill_id_replaced, t1, t2, t3, t4)

def replace_bill_ids(old, new):
    t1 = 0  #keeps track of how many bill id replacements there were
    t2 = 0
    t3 = 0
    t4 = 0
    
    last_bill_number = ""
    last_bill_number_line = 0
    transition_window_list = []
    line_number = 0
    for line in old:
        line_splits = line.lower().rstrip("\n").split("~")
        
        (new_text, current_bill_number, bill_id_replaced, t1, t2, t3, t4) = replace_bill_ids_in_utterance(line_splits[2], last_bill_number, t1, t2, t3, t4)
        
        if (bill_id_replaced):
            if (current_bill_number != last_bill_number):
                transition_window_list.append((last_bill_number_line, line_number))
                last_bill_number = current_bill_number
                last_bill_number_line = line_number
            elif (current_bill_number == last_bill_number):
                last_bill_number_line = line_number
        
        new.write(line_splits[0] + "~" + line_splits[1] + "~" + new_text + "~" + line_splits[3] + "\n")
        line_number += 1
        
    #print("Length of Bill Patterns Replaced\n1: " + str(t1) + "\n2: " + str(t2) + "\n3: " + str(t3) + "\n4: " + str(t4))
    return transition_window_list


# Define add context function (prefix and postfix words in surrounding utterances).
# adds the prefix POST to all utterances n after
# adds the prefix PRE to all utterances n before
# a transition phrase
def add_context(n, cleaned_raw_bill_id_replaced_filename):
    n_range = pd.read_csv(cleaned_raw_bill_id_replaced_filename, sep="~")
    
    transition_text = n_range['text']
    new_transition_text = []

    length = len(n_range)
    
    for i in range(length):
        # get the phrases in the window
        text = ''
        for x in range(-n, n+1):
            # window is within range of the dataframe
            if (i + x >= 0 and i + x < length):
                if (x > 0):
                    text += ' '.join(["POST-" + x for x in transition_text[i+x].split()])
                if (x < 0):
                    text += ' '.join(["PRE-" + x for x in transition_text[i+x].split()])
                else:
                    text += ' ' + transition_text[i+x] + ' '
                    
        new_transition_text.append(text)
    
    print ("Number of new phrases {0}".format(len(new_transition_text)))
    
    n_range.drop(['text'], axis=1, inplace=True)
    n_range['text'] = new_transition_text
    
    n_range.to_csv(cleaned_raw_bill_id_replaced_filename, sep="~", index=False)

def predict_entire_transcript(transcripts, model, count_vect):
    transcripts_test = count_vect.transform(transcripts['text'])
    
    probs = model.predict_proba(transcripts_test)
    preds = [1 if p[1] > THRESHOLD_PROBABILITY else 0 for p in probs]
    
    assert len(preds) == transcripts_test.shape[0]
    
    return preds

def predict_from_naive_bayes(transcript):
    model = pickle.load(open(model_filename, "rb"))
    count_vect = pickle.load(open(count_vectorizer_filename, "rb"))
    #transcript = transcript[transcript["video_id"]==4161]

    prediction_values = predict_entire_transcript(transcript, model, count_vect)
    transcript["prediction"] = prediction_values
    predicted_transitions = transcript[transcript["prediction"]==1]
    transition_dictionary = predicted_transitions[["video_id", "start", "end", "text"]].to_dict(orient="records")
    
    return transition_dictionary

def predict_from_neural_network():
    pass

# new_transcript is a list of dictionaries
def process_raw_data(new_transcript):
	### Determine the raw .srt file that will be processed.

	cleaned_raw_filename = "cleaned_transcript.csv"
	cleaned_raw_bill_id_replaced_filename = cleaned_raw_filename + "2"

	print("...Parsing raw transcript...")

	cleaned_raw = pd.DataFrame(new_transcript)
	cleaned_raw.sort_values(["video_id", "start"]) \
	 .to_csv(cleaned_raw_filename, sep="~", index=False)

	print("...Raw transcript parsed, beginning text formatting and bill replacement...")

	transition_window_list = [] #not currently used, but is available for use

	with open(cleaned_raw_filename, 'r') as old:
	    with open(cleaned_raw_bill_id_replaced_filename, 'w') as new:
	        # consume/write headings
	        h = old.readline()
	        new.write(h)
	            
	        #actually iterate through the file
	        transition_window_list = replace_bill_ids(old, new)

	print("...Text formatted and bills replaced, adding context...")

	add_context(5, cleaned_raw_bill_id_replaced_filename)

	print("...Context added.")

	processed_new_transcript = pd.read_csv(cleaned_raw_bill_id_replaced_filename, sep="~")
	return processed_new_transcript

# returns None if no bill name found, returns bill name if it is found
def extract_bill_name(text):
    print("edited text: ")
    print(textedit.correct_text(list(text)))
    return None


def enhance_dictionary_helper(combined): 
    enhanced_dictionary = []
    
    start_list = list(combined["start"])
    end_list = list(combined["end"])
    video_id_list = list(combined["video_id"])
    original_text_list = list(combined["text_original"])
    modified_text_list = list(combined["text_modified"])
    bill_names_list = list(combined["bill_names"])
    
    found_associated_bill_id = True
    i = 0
    while (i < len(original_text_list)):
        # if the utterance is not a transition utterance, continue
        if (pd.isnull(modified_text_list[i])):
            i += 1
            continue
        
        #otherwise, if there is a bill id immediately associated with the utterance,
        #input it into the dictionary, then continue
        if (len(bill_names_list[i]) > 0):
            enhanced_dictionary.append({
                    'start':start_list[i], 'end':end_list[i],
                    'video_id':video_id_list[i], 'text':original_text_list[i],
                    'suggested_bill':bill_names_list[i][0]})
            i += 1
            continue
        
        #otherwise, scan all table rows until the next transition utterance looking for
        #an associated bill id
        j = i+1
        while(j < len(original_text_list) and pd.isnull(modified_text_list[j])):
            if (len(bill_names_list[j]) > 0):
                enhanced_dictionary.append({
                    'start':start_list[i], 'end':end_list[i],
                    'video_id':video_id_list[i], 'text':original_text_list[i],
                    'suggested_bill':bill_names_list[j][0]})
                i = j+1
                found_associated_bill_id = False
                break
            j += 1
            
        if (not found_associated_bill_id):
            found_associated_bill_id = True
            continue
        
        #at this point, the transition utterance has no associated bill ids in the window
        #so we likely want to disregard the transition as a possibility
        #for now we simply enter it with the suggested_bill field having value 'NONE'
        enhanced_dictionary.append({
                    'start':start_list[i], 'end':end_list[i],
                    'video_id':video_id_list[i], 'text':original_text_list[i],
                    'suggested_bill':'NONE'})
        i = j
            
    return enhanced_dictionary


def enhance_dictionary(original_transcript, transition_dictionary):
    #transition_dictitionary format: [{video_id, start, end, text}]
    #original transcript format new_transcript = [{'start':'1', 'end':'9', 'text':'we are starting', 'video_id':'1'}]
    
    original = pd.DataFrame.from_dict(original_transcript)
    transitions = pd.DataFrame.from_dict(transition_dictionary)
    
    #combined = pd.merge(original, transitions, on=["video_id", "start", "end"], suffixes=("_original", "_modified"))
    combined = pd.merge(original, transitions, how='left', on=["video_id", "start", "end"], suffixes=("_original", "_modified")) 
    substituted_original = textedit.correct_text(list(combined["text_original"]))
    
    bill_names_in_text = []
    for text in substituted_original:
        bill_names_in_text.append(re.findall("[a-zA-Z]+\\s*[0-9]+", text))
        
    combined["bill_names"] = bill_names_in_text
    
    return enhance_dictionary_helper(combined)
    
    
# given a list of dictionaries, return a 
def predict(new_transcript):
    processed_new_transcript = process_raw_data(new_transcript)

    transition_dictionary = {}

    if (model_type == "NB"):
        transition_dictionary = predict_from_naive_bayes(processed_new_transcript)

    elif (model_type == "NN"):
        raise Exception("Neural network not supported yet.")
        predict_from_neural_network()

    else:
        raise Exception("Model type not defined.")

    enhanced_dictionary = enhance_dictionary(new_transcript, transition_dictionary)
    return enhanced_dictionary
    

'''
def main():
    new_transcript = [{'start':1, 'end':9, 'text':'we are starting', 'video_id':1}, 
     {'start':9, 'end':12, 'text':'we keep going on assembly bill three four five six', 'video_id':1},
     {'start':14, 'end':20, 'text':'is this a', 'video_id':1},
     {'start':21, 'end':22, 'text':'what do you think of senate bill 134?', 'video_id':1}]
    transition_dictionary = predict(new_transcript)
    enhanced_dictionary = enhance_dictionary(new_transcript, transition_dictionary)
    print(enhanced_dictionary)

if __name__ == "__main__":
    main()
'''
