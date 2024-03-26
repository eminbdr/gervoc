from collections import defaultdict
import spacy
from tqdm import tqdm
import sys
from collections import defaultdict
import dill
import os

nlp = spacy.load("de_core_news_sm", disable=["ner", "tagger", "parser"])

pos_dict = defaultdict(lambda: defaultdict(int))
total_lines = 41612280
path = "C://Users//muham//Downloads//de.txt//de.txt"

#Only processed 700000 lines due to low computational power


with open("C://Users//muham//Downloads//de.txt//de.txt", "r", encoding="utf-8") as file:
    if os.path.exists("pos_dict_raw.pkl"):
        pos_dict = dill.load(open("pos_dict_raw.pkl", "rb")) 

    for line_num, line in enumerate(tqdm(file, total=total_lines, desc="Processing file")):
        if line_num < 700000:
            continue
        doc = nlp(line.strip())
        for token in doc:
            if not token.is_stop and not token.is_punct:  # Check if token is not a stop word or punctuation
                # Use token.lemma_ to get the lemma (root) of the token
                # Use token.pos_ to get the type (part of speech) of the token
                pos_dict[token.pos_][(token.lemma_, token.pos_)] += 1  # Count frequency
        
        # Check for user input after processing every 10000 lines
        if line_num % 1000000 == 0:
            user_input = input("Press 'q' to quit or press Enter to continue: ")
            if user_input.lower() == 'q':
                sys.exit("Execution terminated by user.")


data = dill.load(open("pos_dict_raw.pkl", "rb"))
verb =dict(sorted(list(data["VERB"].items()), key=lambda item: item[1], reverse=True))
noun =dict(sorted(list(data["NOUN"].items()), key=lambda item: item[1], reverse=True))
adj =dict(sorted(list(data["ADJ"].items()), key=lambda item: item[1], reverse=True))
adv =dict(sorted(list(data["ADV"].items()), key=lambda item: item[1], reverse=True))