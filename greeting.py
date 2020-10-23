#!/usr/bin/env python
# coding: utf-8
"""Using the parser to recognise your own semantics

spaCy's parser component can be trained to predict any type of tree
structure over your input text. You can also predict trees over whole documents
or chat logs, with connections between the sentence-roots used to annotate
discourse structure. In this example, we'll build a message parser for a common
"chat intent": finding local businesses. Our message semantics will have the
following types of relations: ROOT, PLACE, QUALITY, ATTRIBUTE, TIME, LOCATION.

"show me the best hotel in berlin"
('show', 'ROOT', 'show')
('best', 'QUALITY', 'hotel') --> hotel with QUALITY best
('hotel', 'PLACE', 'show') --> show PLACE hotel
('berlin', 'LOCATION', 'hotel') --> hotel with LOCATION berlin

Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from spacy import displacy


# training data: texts, heads and dependency labels
# for no relation, we simply chose an arbitrary dependency label, e.g. '-'
TRAIN_DATA = [
    (
        #0  1   2   3
        "hi how are you",
        {
            "heads": [0, 0, 3, 1],  # index of token head
            "deps": ["ROOT", "QUESTION", "STATE", "TARGET"],
        },
    ),
    (
        #0  1   2   3   4
        "hi how are you feeling",
        {
            "heads": [0, 0, 3, 1, 3],  # index of token head
            "deps": ["ROOT", "QUESTION", "STATE", "TARGET", "ATTRIBUTE"],
        },
    ),
    (
        #0   1   2   3   4   5
        "hey bot how are you today",
        {
            "heads": [0, 2, 0, 1, 2, 2],  # index of token head
            "deps": ["ROOT", "TARGET", "QUESTION", "STATE", "TARGET", "-"],
        },
    ),
    (
        #0     1  2
        "hello mr bot",
        {
            "heads": [0, 0, 0],
            "deps": ["QUESTION", "-", "TARGET"],
        },
    ),
    (
        #0   1   2   3
        "how are you feeling",
        {
            "heads": [0, 2, 0, 2],
            "deps": ["QUESTION", "STATE", "TARGET", "ATTRIBUTE"],
        },
    ),
    (
        #0   1  2   3  4
        "how do you do chatbot",
        {
            "heads": [0, 2, 0, 2, 0],
            "deps": ["QUESTION", "-", "TARGET", "STATE", "TARGET"],
        },
    ),
    # hello there
    (
        #0     1
        "hello there",
        {
            "heads": [0, 0],
            "deps": ["ROOT", "-"],
        },
    ),
    # # hi there
    # (
    #     "hi there",
    #     {
    #         "heads": [0, 0],
    #         "deps": ["ROOT", "-"],
    #     },
    # ),
    # # greetings
    # (
    #     "greetings",
    #     {
    #         "heads": [0],
    #         "deps": ["ROOT"],
    #     },
    # ),
    # # howdy
    # (
    #     "howdy",
    #     {
    #         "heads": [0],
    #         "deps": ["ROOT"],
    #     },
    # ),
    # # hey
    # (
    #     "hey",
    #     {
    #         "heads": [0],
    #         "deps": ["ROOT"],
    #     },
    # ),
    # # hihi
    # (
    #     "hihi",
    #     {
    #         "heads": [0],
    #         "deps": ["ROOT"],
    #     },
    # ),
    (
        "hmmm hello there larry",
        {
            "heads": [1, 1, 1, 1],
            "deps": ["-", "ROOT", "-", "TARGET"],
        },
    ),


    # (
    #     "find a cafe with great wifi",
    #     {
    #         "heads": [0, 2, 0, 5, 5, 2],  # index of token head
    #         "deps": ["ROOT", "-", "PLACE", "-", "QUALITY", "ATTRIBUTE"],
    #     },
    # ),
    # (
    #     "find a hotel near the beach",
    #     {
    #         "heads": [0, 2, 0, 5, 5, 2],
    #         "deps": ["ROOT", "-", "PLACE", "QUALITY", "-", "ATTRIBUTE"],
    #     },
    # ),
    # (
    #     "find me the closest gym that's open late",
    #     {
    #         "heads": [0, 0, 4, 4, 0, 6, 4, 6, 6],
    #         "deps": [
    #             "ROOT",
    #             "-",
    #             "-",
    #             "QUALITY",
    #             "PLACE",
    #             "-",
    #             "-",
    #             "ATTRIBUTE",
    #             "TIME",
    #         ],
    #     },
    # ),
    # (
    #     "show me the cheapest store that sells flowers",
    #     {
    #         "heads": [0, 0, 4, 4, 0, 4, 4, 4],  # attach "flowers" to store!
    #         "deps": ["ROOT", "-", "-", "QUALITY", "PLACE", "-", "-", "PRODUCT"],
    #     },
    # ),
    # (
    #     "find a nice restaurant in london",
    #     {
    #         "heads": [0, 3, 3, 0, 3, 3],
    #         "deps": ["ROOT", "-", "QUALITY", "PLACE", "-", "LOCATION"],
    #     },
    # ),
    # (
    #     "show me the coolest hostel in berlin",
    #     {
    #         "heads": [0, 0, 4, 4, 0, 4, 4],
    #         "deps": ["ROOT", "-", "-", "QUALITY", "PLACE", "-", "LOCATION"],
    #     },
    # ),
    # (
    #     "find a good italian restaurant near work",
    #     {
    #         "heads": [0, 4, 4, 4, 0, 4, 5],
    #         "deps": [
    #             "ROOT",
    #             "-",
    #             "QUALITY",
    #             "ATTRIBUTE",
    #             "PLACE",
    #             "ATTRIBUTE",
    #             "LOCATION",
    #         ],
    #     },
    # ),
]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, output_dir=None, n_iter=15):
    """Load the model, set up the pipeline and train the parser."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        # nlp = spacy.blank("en")  # create blank Language class
        nlp = spacy.load("en_core_web_sm")
        print("Created blank 'en' model")

    # We'll use the built-in dependency parser class, but we want to create a
    # fresh instance – just in case.
    if "parser" in nlp.pipe_names:
        nlp.remove_pipe("parser")
    parser = nlp.create_pipe("parser")
    nlp.add_pipe(parser, first=True)

    for text, annotations in TRAIN_DATA:
        for dep in annotations.get("deps", []):
            parser.add_label(dep)

    pipe_exceptions = ["parser", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    # other_pipes = []
    with nlp.disable_pipes(*other_pipes):  # only train parser
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, losses=losses)
            print("Losses", losses)

    # test the trained model
    test_model(nlp)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        test_model(nlp2)


greetings = ["hi", "hello", "hey", "greetings"]
questions = ["who", "what", "how", "when", "where", "why"]

targets_bot = ["bot", "self", "chatbot", "larry", "you", "world"]

greetings_responses = [
    "Hi!",
    "Heya!",
    "Hello friendly human.",
    "Hi hi.",
    "Oh hey there!",
]

questions_state_responses = [
    "I'm doing fine, and you?",
    "I'm alright. How about you?",
    "I'm feeling great! Really inspired. But how are you?",
    "I'm fine, thanks for asking! And you?",
    "I'm really sleepy, and you?",
]

def has_labels(doc, *labels):
    doc_labels = [t.dep_ for t in doc]
    return all(elem in doc_labels for elem in labels)


def test_model(nlp):
    texts = [
        # "find a hotel with good wifi",
        # "find me the cheapest gym near work",
        # "show me the best hotel in berlin",
        # "find a good hotel please",
        "hiiiiiiii",
        "holi",
        "HELLO!!!",
        "hi bot",
        "hi how are you",
        "hello world",
        "Hey bot",
    ]


    docs = nlp.pipe(texts)
    for doc in docs:
        print(doc.text)

        # Print the following values:
        # - Original text
        # - Dependency label
        # - Head text
        # - Lemma?
        # - Part of s
        print([(t.text, t.dep_, t.head.text, t.lemma_, t.pos_) for t in doc if t.dep_ != "-"])

        # Render a dependency tree in html and save to one file per example
        html = displacy.render(doc, style='dep', page=True)
        file_name = '-'.join([w.text for w in doc if not w.is_punct]) + ".html"
        output_path = Path("results\\" + file_name)
        output_path.open("w", encoding="utf-8").write(html)

        # Ignore "-"s
        # doc = [t for t in doc if t.dep_ != "-"]

        # Print symbols
        syms = [t for t in doc if t.pos_ == "SYM" or t.pos_ == "PUNCT"]
        print(f"syms: {syms}")

        # Remove punctuation
        doc = [t for t in doc if t.pos_ != "PUNCT"]

        # Create a dictionary, grouping tokens by their dependency label
        label_dict = {t.dep_ : t for t in doc}
        print("label_dict")
        print(label_dict)

        # Allow multiple responses
        responses = []

        # Answer greetings
        for t in doc:
            if t.dep_ == "ROOT":
                if t.lemma_.lower() in greetings:
                    responses += [random.choice(greetings_responses)]

        # Answer a basic question of "how are you"
        if has_labels(doc, "ROOT", "STATE", "TARGET"):
            # Check if it's a question
            if "QUESTION" in label_dict or label_dict["ROOT"].lemma_.lower() in questions:
                # It's a question, see who's the target
                if label_dict["TARGET"].lemma_.lower() in targets_bot:
                    responses += [random.choice(questions_state_responses)]
        
        if responses:
            print("RESPONSE:")
            print(" ".join(responses))

        print("---------------------------------------\n\n")
            





if __name__ == "__main__":
    plac.call(main)

    # Expected output:
    # find a hotel with good wifi
    # [
    #   ('find', 'ROOT', 'find'),
    #   ('hotel', 'PLACE', 'find'),
    #   ('good', 'QUALITY', 'wifi'),
    #   ('wifi', 'ATTRIBUTE', 'hotel')
    # ]
    # find me the cheapest gym near work
    # [
    #   ('find', 'ROOT', 'find'),
    #   ('cheapest', 'QUALITY', 'gym'),
    #   ('gym', 'PLACE', 'find'),
    #   ('near', 'ATTRIBUTE', 'gym'),
    #   ('work', 'LOCATION', 'near')
    # ]
    # show me the best hotel in berlin
    # [
    #   ('show', 'ROOT', 'show'),
    #   ('best', 'QUALITY', 'hotel'),
    #   ('hotel', 'PLACE', 'show'),
    #   ('berlin', 'LOCATION', 'hotel')
    # ]
