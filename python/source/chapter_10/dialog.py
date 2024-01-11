import re
import abc
import operator
import collections


class Dialog(abc.ABC):
    """
    A dialog listens for utterances, parses and interprets them, then updates
    its internal state. It can then formulate a response on demand.
    """

    def listen(self, text, need_response=True, **kwargs):
        """
        A text utterance is passed in and parsed. It is then passed to the
        interpret method to determine how to respond. If a response is
        requested, the respond method is used to generate a text response
        based on the most recent input and the current Dialog state.
        """
        # Parse the input
        sents = self.parse(text)

        # Interpret the input
        sents, confidence, kwargs = self.interpret(sents, **kwargs)

        # Determine the response
        response = (self.respond(sents, confidence, **kwargs)
                    if need_response else None)

        # Return initiative
        return response, confidence

    @abc.abstractmethod
    def parse(self, text):
        """
        Every dialog may need its own parsing strategy, some dialogs may need
        dependency vs. constituency parses, others may simply require regular
        expressions or chunkers.
        """
        return []

    @abc.abstractmethod
    def interpret(self, sents, **kwargs):
        """
        Interprets the utterance passed in as a list of parsed sentences,
        updates the internal state of the dialog, computes a confidence of the
        interpretation. May also return arguments specific to the response
        mechanism.
        """
        return sents, 0.0, kwargs

    @abc.abstractmethod
    def respond(self, sents, confidence, **kwargs):
        """
        Creates a response given the input utterances and the current state of
        the dialog, along with any arguments passed in from the listen or the
        interpret methods.
        """
        return None


class SimpleConversation(Dialog, collections.abc.Sequence):
    """
    This is the most simple version of a conversation.
    """

    def __init__(self, dialogs):
        self._dialogs = dialogs

    def __getitem__(self, idx):
        return self._dialogs[idx]

    def __len__(self):
        return len(self._dialogs)

    def listen(self, text, need_response=True, **kwargs):
        """
        Simply return the best confidence response
        """
        responses = [dialog.listen(text, need_response, **kwargs)
                     for dialog in self._dialogs]

        # Responses is a list of (response, confidence) pairs
        return max(responses, key=operator.itemgetter(1))

    def parse(self, text):
        """
        Returns parses for all internal dialogs for debugging
        """
        return [dialog.parse(text)
                for dialog in self._dialogs]

    def interpret(self, sents, **kwargs):
        """
        Returns interpretations for all internal dialogs for debugging
        """
        return [dialog.interpret(sents, **kwargs)
                for dialog in self._dialogs]

    def respond(self, sents, confidence, **kwargs):
        """
        Returns responses for all internal dialogs for debugging
        """
        return [dialog.respond(sents, confidence, **kwargs)
                for dialog in self._dialogs]


class Greeting(Dialog):
    """
    Keeps track of the participants entering or leaving the conversation and
    responds with appropriate salutations. This is an example of a rules based
    system that keeps track of state and uses regular expressions and logic to
    handle the dialog.
    """

    PATTERNS = {
        'greeting': r'hello|hi|hey|good morning|good evening',
        'introduction': r'my name is ([a-z\-\s]+)',
        'goodbye': r'goodbye|bye|ttyl',
        'rollcall': r'roll call|who\'s here?'
    }

    def __init__(self, participants=None):
        # Participants is a map of user name to real name
        self.participants = {}

        if participants is not None:
            for participant in participants:
                self.participants[participant] = None

        # Compile regular expressions
        self._patterns = {
            key: re.compile(pattern, re.I)
            for key, pattern in self.PATTERNS.items()
        }

    def parse(self, text):
        """
        Applies all regular expressions to the text to find matches.
        """
        return {
            key: match
            for key, pattern in self._patterns.items()
            if (match := pattern.search(text))
            and match is not None
        }

    def interpret(self, sents, **kwargs):
        """
        Takes in parsed matches and determines if the message is an enter,
        exit, or name change.
        """
        # Can't do anything with no matches
        if len(sents) == 0:
            return sents, 0.0, kwargs

        # Get username from the participants
        user = kwargs.get('user', None)

        # Determine if an introduction has been made
        if 'introduction' in sents:
            # Get the name from the utterance
            name = sents['introduction'].groups()[0]
            user = user or name.lower()

            # Determine if name has changed
            if (user not in self.participants
                or self.participants[user] != name):
                kwargs['name_changed'] = True

            # Update the participants
            self.participants[user] = name
            kwargs['user'] = user

        # Determine if a greeting has been made
        if 'greeting' in sents:
            # If we don't have a name for the user
            if user not in self.participants:
                kwargs['request_introduction'] = True

        # Determine if goodbye has been made
        if 'goodbye' in sents and user is not None:
            # Remove participant
            self.participants.pop(user)
            kwargs.pop('user', None)

        # If we've seen anything we're looking for, we're pretty confident
        return sents, 1.0, kwargs

    def respond(self, sents, confidence, **kwargs):
        """
        Gives a greeting or a goodbye depending on what's appropriate.
        """
        if confidence == 0:
            return None

        name = self.participants.get(kwargs.get('user', None), None)
        name_changed = kwargs.get('name_changed', False)
        request_introduction = kwargs.get('request_introduction', False)

        if 'greeting' in sents or 'introduction' in sents:
            if request_introduction:
                return "Hello, what is your name?"
            else:
                return "Hello, {}!".format(name)

        if 'goodbye' in sents:
            return "Talk to you later!"

        if 'rollcall' in sents:
            people = list(self.participants.values())

            if len(people) > 1:
                roster = ", ".join(people[:-1])
                roster += " and {}.".format(people[-1])
                return "Currently in the conversation are " + roster
            elif len(people) == 1:
                return "It's just you and me right now, {}.".format(name)
            else:
                return "So lonely in here by myself ... wait who is that?"

        raise Exception(
            "expected response to be returned, but could not find rule"
        )
