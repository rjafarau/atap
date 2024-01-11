import pytest

from dialog import Dialog, Greeting


class TestDialogClass(object):
    """
    Tests for the Dialog class
    """

    @pytest.mark.parametrize("text", ["Gobbledeguk", "Gibberish", "Wingdings"])
    def test_dialog_abc(self, text):
        """
        Test the Dialog ABC and the listen method
        """
        class SampleDialog(Dialog):

            def parse(self, text):
                return []

            def interpret(self, sents):
                return sents, 0.0, {}

            def respond(self, sents, confidence):
                return None

        sample = SampleDialog()
        reply, confidence = sample.listen(text)
        assert confidence == 0.0
        assert reply is None


class TestGreetingClass(object):
    """
    Test expected input and responses for the Greeting dialog
    """

    @pytest.mark.parametrize("text", ["Hello!", "hello", 'hey', 'hi'])
    @pytest.mark.parametrize("user", ["jay", None], ids=["w/ user", "w/o user"])
    def test_greeting_intro(self, user, text):
        """
        Test that an initial greeting requests an introduction
        """
        g = Greeting()
        reply, confidence = g.listen(text, user=user)
        assert confidence == 1.0
        assert reply is not None
        assert reply == "Hello, what is your name?"

    @pytest.mark.xfail(reason="a case that must be handled")
    @pytest.mark.parametrize("text", ["My name is Jake", "Hello, I'm Jake."])
    @pytest.mark.parametrize("user", ["jkm", None], ids=["w/ user", "w/o user"])
    def test_initial_intro(self, user, text):
        """
        Test an initial introduction without greeting
        """
        g = Greeting()
        reply, confidence = g.listen(text, user=user)
        assert confidence == 1.0
        assert reply is not None
        assert reply == "Hello, Jake!"

        if user is None:
            user = 'jake'

        assert user in g.participants
        assert g.participants[user] == 'Jake'
