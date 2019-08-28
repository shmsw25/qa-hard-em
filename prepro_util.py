import numpy as np
import unicodedata
import tokenization
from collections import Counter, defaultdict

class SquadExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 paragraph_indices=None,
                 orig_answer_text=None,
                 all_answers=None,
                 start_position=None,
                 end_position=None,
                 switch=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.paragraph_indices = paragraph_indices
        self.orig_answer_text = orig_answer_text
        self.all_answers=all_answers
        self.start_position = start_position
        self.end_position = end_position
        self.switch = switch

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = "question: "+self.question_text
        return s

class InputFeatures(object):

    def __init__(self,
                 unique_id,
                 example_index,
                 paragraph_index=None,
                 doc_span_index=None,
                 doc_tokens=None,
                 tokens=None,
                 token_to_orig_map=None,
                 token_is_max_context=None,
                 input_ids=None,
                 input_mask=None,
                 segment_ids=None,
                 start_position=None,
                 end_position=None,
                 switch=None,
                 answer_mask=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.paragraph_index = paragraph_index
        self.doc_span_index = doc_span_index
        self.doc_tokens = doc_tokens
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.switch = switch
        self.answer_mask = answer_mask


def _run_strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
      cat = unicodedata.category(char)
      if cat == "Mn":
        continue
      output.append(char)
    return "".join(output)


def find_span_from_text(context, tokens, answer):
    assert answer in context

    offset = 0
    spans = []
    scanning = None
    process = []

    for i, token in enumerate(tokens):
        token = token.replace(' ##', '').replace('##', '')
        while context[offset:offset+len(token)]!=token:
            offset += 1
            if offset >= len(context):
                break
        if scanning is not None:
            end = offset + len(token)
            if answer.startswith(context[scanning[-1][-1]:end]):
                if context[scanning[-1][-1]:end] == answer:
                    span = (scanning[0][0], i, scanning[0][1])
                    spans.append(span)
                elif len(context[scanning[-1][-1]:end]) >= len(answer):
                    scanning = None
            else:
                scanning = None
        if scanning is None and answer.startswith(token):
            if token == answer:
                spans.append((i, i, offset))
            if token != answer:
                scanning = [(i, offset)]
        offset += len(token)
        if offset >= len(context):
            break
        process.append((token, offset, scanning, spans))

    answers = []

    for word_start, word_end, span in spans:
        assert context[span:span+len(answer)]==answer or ''.join(tokens[word_start:word_end+1]).replace('##', '')!=answer.replace(' ', '')
        answers.append({'text': answer, 'answer_start': span, 'word_start': word_start, 'word_end': word_end})

    return answers

def detect_span(_answers, context, doc_tokens, char_to_word_offset):
    orig_answer_texts = []
    start_positions = []
    end_positions = []
    switches = []

    answers = []
    for answer in _answers:
        answers += find_span_from_text(context, doc_tokens, answer['text'])

    for answer in answers:
        orig_answer_text = answer["text"]
        answer_offset = answer["answer_start"]
        answer_length = len(orig_answer_text)

        switch = 0
        if 'word_start' in answer  and 'word_end' in answer:
            start_position = answer['word_start']
            end_position = answer['word_end']
        else:
            start_position = char_to_word_offset[answer_offset]
            end_position = char_to_word_offset[answer_offset + answer_length - 1]
        # Only add answers where the text can be exactly recovered from the
        # document. If this CAN'T happen it's likely due to weird Unicode
        # stuff so we will just skip the example.
        #
        # Note that this means for training mode, every example is NOT
        # guaranteed to be preserved.
        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)]).replace(' ##', '').replace('##', '')
        cleaned_answer_text = " ".join(
            tokenization.whitespace_tokenize(orig_answer_text))
        if actual_text.replace(' ', '').find(cleaned_answer_text.replace(' ', '')) == -1:
            print ("Could not find answer: '%s' vs. '%s'" % (actual_text, cleaned_answer_text))

        orig_answer_texts.append(orig_answer_text)
        start_positions.append(start_position)
        end_positions.append(end_position)
        switches.append(switch)


    return orig_answer_texts, switches, start_positions, end_positions

