import pprint

import nltk
import spacy


def plot_displacy_tree(model, sent):
    doc = model(sent)
    spacy.displacy.render(doc, style='dep')


def spacy_tree(model, sent):
    """
    Get the SpaCy dependency tree structure
    :param sent: string
    :return: None
    """
    doc = model(sent)
    pprint.pprint(doc.to_json())


def nltk_spacy_tree(model, sent):
    """
    Visually inspect the SpaCy dependency tree with nltk.tree
    :param sent: string
    :return: None
    """
    doc = model(sent)

    def token_format(token):
        return "_".join([token.orth_, token.tag_, token.dep_])

    def to_nltk_tree(node):
        if node.n_lefts + node.n_rights > 0:
            return nltk.Tree(token_format(node),
                             [to_nltk_tree(child)
                              for child in node.children])
        else:
            return token_format(node)

    return [to_nltk_tree(sent.root) for sent in doc.sents][0]


def question_type(model, sent):
    """
    Try to identify whether the question is about measurements,
    recipes, or not a question.
    :param sent: string
    :return: str response type
    """
    doc = model(sent)

    noun_tags = {'NN', 'NNS', 'NNP', 'NNPS'}
    nouns = [token.orth_
             for sent in doc.sents
             for token in sent
             if token.tag_ in noun_tags]
    for sent in doc.sents:
        for token in sent:
            # Find wh-adjective and wh-adverb phrases
            if token.tag_ == 'WRB':
                if token.nbor().tag_ == 'JJ':
                    return ("quantity", nouns)
            # Find wh-noun phrases
            elif token.tag_ == 'WP':
                # Use pre-trained clusters to return recipes
                return ("recipe", nouns)
    # Todo: try to be conversational using our n-gram language generator?
    return ("default", nouns)
