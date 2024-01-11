import json

import nltk
import spacy
import inflect
import humanize

from dialog import Dialog


class Converter(Dialog):
    """
    Answers questions about converting units
    """

    def __init__(self, conversion_path, stemmer, parser):
        with open(conversion_path, 'r') as f:
            self.metrics = json.load(f)
        self.inflect = inflect.engine()
        self.stemmer = stemmer
        self.parser = parser

    def parse(self, text):
        parse = self.parser(text)
        return parse

    def interpret(self, sents, **kwargs):
        measures = []
        confidence = 0
        results = {}
        # Make sure there are wh-adverb phrases
        if 'WRB' in [token.tag_
                     for sent in sents.sents
                     for token in sent]:
            # If so, increment confidence & traverse sents
            confidence += .2
            for sent in sents.sents:
                for token in sent:
                    # Store nouns as target measures
                    if token.tag_ in ['NN', 'NNS']:
                        measures.append(token.orth_)
                    # Store numbers as target quantities
                    elif token.tag_ in ['CD']:
                        results['quantity'] = token.orth_

            # If both source and destination measures are provided...
            if len(measures) == 2:
                confidence += .4
                # Stem source and dest to remove pluralization
                results['dst'], results['src'] = (
                    tuple(map(self.stemmer.stem, measures))
                )

                # Check to see if they correspond to our lookup table
                if results['src'] in self.metrics:
                    confidence += .2
                    if results['dst'] in self.metrics[results['src']]:
                        confidence += .2

        return results, confidence, kwargs

    def convert(self, src, dst, quantity=1.0):
        """
        Converts from the source unit to the dest unit for the given quantity
        of the source unit.
        """
        # Check that we can convert
        if dst not in self.metrics:
            raise KeyError(f"cannot convert to '{dst}' units")
        if src not in self.metrics[dst]:
            raise KeyError(f"cannot convert from '{src}' to '{dst}'")

        return self.metrics[dst][src] * float(quantity), src, dst

    def round(self, num):
        num = round(float(num), 4)
        return int(num) if num.is_integer() else num

    def pluralize(self, noun, num):
        return self.inflect.plural_noun(noun, num)

    def numericalize(self, amt):
        if 1e2 < amt < 1e6:
            return humanize.intcomma(int(amt))
        elif amt >= 1e6:
            return humanize.intword(int(amt))
        elif isinstance(amt, int) or amt.is_integer():
            return humanize.apnumber(int(amt))
        else:
            return humanize.fractional(amt)

    def respond(self, sents, confidence, **kwargs):
        """
        Response makes use of the humanize and inflect libraries to produce
        much more human understandable results.
        """
        if confidence < .5:
            return "I'm sorry, I don't know that one."

        try:
            quantity = sents.get('quantity', 1)
            amount, src, dst = self.convert(**sents)

            # Perform numeric rounding
            amount = self.round(amount)
            quantity = self.round(quantity)

            # Pluralize
            src = self.pluralize(src, quantity)
            dst = self.pluralize(dst, amount)
            verb = self.inflect.plural_verb('is', amount)

            # Numericalize
            quantity = self.numericalize(quantity)
            amount = self.numericalize(amount)

            return f'There {verb} {amount} {dst} in {quantity} {src}.'

        except KeyError as e:
            return "I'm sorry I {}".format(str(e))
