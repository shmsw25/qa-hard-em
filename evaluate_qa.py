import os
import pickle as pkl
import json
import collections
import math
import six
import time
import numpy as np
import tokenization
from collections import defaultdict
from tqdm import tqdm
from evaluation_script import normalize_answer, f1_score, exact_match_score

rawResult = collections.namedtuple("RawResult",
                                  ["unique_id", "start_logits", "end_logits"])
_NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
           "NbestPrediction", ["text", "logit", "no_answer_logit"])


def write_predictions(logger, all_examples, all_features, all_results, n_best_size,
                      do_lower_case, output_prediction_file,
                      output_nbest_file, verbose_logging,
                      write_prediction=True, n_paragraphs=None):

    """Write final predictions to the json file."""

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
       "PrelimPrediction",
       ["paragraph_index", "feature_index", "start_index", "end_index", "logit", "no_answer_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    if verbose_logging:
        all_examples = tqdm(enumerate(all_examples))
    else:
        all_examples = enumerate(all_examples)

    for (example_index, example) in all_examples:
        features = example_index_to_features[example_index]
        if len(features)==0 and n_paragraphs is None:
            pred = _NbestPrediction(
                        text="empty",
                        logit=-1000,
                        no_answer_logit=1000)
            all_predictions[example.qas_id] = ("empty", example.all_answers)
            all_nbest_json[example.qas_id] = [pred]
            continue

        prelim_predictions = []
        yn_predictions = []

        if n_paragraphs is None:
            results = sorted(enumerate(features),
                         key=lambda f: unique_id_to_result[f[1].unique_id].switch[3])[:1]
        else:
            results = enumerate(features)
        for (feature_index, feature) in results:
            result = unique_id_to_result[feature.unique_id]
            scores = []
            start_logits = result.start_logits[:len(feature.tokens)]
            end_logits = result.end_logits[:len(feature.tokens)]
            for (i, s) in enumerate(start_logits):
                for (j, e) in enumerate(end_logits[i:i+10]):
                    scores.append(((i, i+j), s+e))

            scores = sorted(scores, key=lambda x: x[1], reverse=True)

            cnt = 0
            for (start_index, end_index), score in scores:
                if start_index >= len(feature.tokens):
                    continue
                if end_index >= len(feature.tokens):
                    continue
                if start_index not in feature.token_to_orig_map:
                    continue
                if end_index not in feature.token_to_orig_map:
                    continue
                if not feature.token_is_max_context.get(start_index, False):
                    continue
                if end_index < start_index:
                    continue
                prelim_predictions.append(
                   _PrelimPrediction(
                       paragraph_index=feature.paragraph_index,
                       feature_index=feature_index,
                       start_index=start_index,
                       end_index=end_index,
                       logit=-result.switch[3], #score,
                       no_answer_logit=result.switch[3]))
                if n_paragraphs is None:
                    if write_predictions and len(prelim_predictions)>=n_best_size:
                        break
                    elif not write_predictions:
                        break
                cnt += 1

        prelim_predictions = sorted(
                prelim_predictions,
                key=lambda x: x.logit,
                reverse=True)
        no_answer_logit = result.switch[3]

        def get_nbest_json(prelim_predictions):

            seen_predictions = {}
            nbest = []
            for pred in prelim_predictions:
                if len(nbest) >= n_best_size:
                    break

                if pred.start_index == pred.end_index == -1:
                    final_text = "yes"
                elif pred.start_index == pred.end_index == -2:
                    final_text = "no"
                else:
                    feature = features[pred.feature_index]

                    tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                    orig_doc_start = feature.token_to_orig_map[pred.start_index]
                    orig_doc_end = feature.token_to_orig_map[pred.end_index]
                    orig_tokens = feature.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                    tok_text = " ".join(tok_tokens)

                    # De-tokenize WordPieces that have been split off.
                    tok_text = tok_text.replace(" ##", "")
                    tok_text = tok_text.replace("##", "")

                    # Clean whitespace
                    tok_text = tok_text.strip()
                    tok_text = " ".join(tok_text.split())
                    orig_text = " ".join(orig_tokens)

                    final_text = get_final_text(tok_text, orig_text, do_lower_case, \
                                                logger, verbose_logging)


                if final_text in seen_predictions:
                    continue

                nbest.append(
                    _NbestPrediction(
                        text=final_text,
                        logit=pred.logit,
                        no_answer_logit=no_answer_logit))

            # In very rare edge cases we could have no valid predictions. So we
            # just create a nonce prediction in this case to avoid failure.
            if not nbest:
                nbest.append(
                _NbestPrediction(text="empty", logit=0.0, no_answer_logit=no_answer_logit))

            assert len(nbest) >= 1

            total_scores = []
            for entry in nbest:
                total_scores.append(entry.logit)

            probs = _compute_softmax(total_scores)
            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = collections.OrderedDict()
                output['text'] = entry.text
                output['probability'] = probs[i]
                output['logit'] = entry.logit
                output['no_answer_logit'] = entry.no_answer_logit
                nbest_json.append(output)

            assert len(nbest_json) >= 1
            return nbest_json
        if n_paragraphs is None:
            nbest_json = get_nbest_json(prelim_predictions)
            all_predictions[example.qas_id] = (nbest_json[0]["text"], example.all_answers)
            all_nbest_json[example.qas_id] = nbest_json
        else:
            all_predictions[example.qas_id] = []
            all_nbest_json[example.qas_id] = []
            for n in n_paragraphs:
                nbest_json = get_nbest_json([pred for pred in prelim_predictions if \
                                             pred.paragraph_index<n])
                all_predictions[example.qas_id].append(nbest_json[0]["text"])

    if write_prediction:
        logger.info("Writing predictions to: %s" % (output_prediction_file))
        logger.info("Writing nbest to: %s" % (output_nbest_file))

        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")

        with open(output_nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if n_paragraphs is None:
        f1s, ems = [], []
        for prediction, groundtruth in all_predictions.values():
            if len(groundtruth)==0:
                f1s.append(0)
                ems.append(0)
                continue
            f1s.append(max([f1_score(prediction, gt)[0] for gt in groundtruth]))
            ems.append(max([exact_match_score(prediction, gt) for gt in groundtruth]))
        final_f1, final_em = np.mean(f1s), np.mean(ems)
    else:
        f1s, ems = [[] for _ in n_paragraphs], [[] for _ in n_paragraphs]
        for predictions in all_predictions.values():
            groundtruth = predictions[-1]
            predictions = predictions[:-1]
            if len(groundtruth)==0:
                for i in range(len(n_paragraphs)):
                    f1s[i].append(0)
                    ems[i].append(0)
                continue
            for i, prediction in enumerate(predictions):
                f1s[i].append(max([f1_score(prediction, gt)[0] for gt in groundtruth]))
                ems[i].append(max([exact_match_score(prediction, gt) for gt in groundtruth]))
        for n, f1s_, ems_ in zip(n_paragraphs, f1s, ems):
            logger.info("n=%d\tF1 %.2f\tEM %.2f"%(n, np.mean(f1s_)*100, np.mean(ems_)*100))
        final_f1, final_em = np.mean(f1s[-1]), np.mean(ems[-1])
    return final_em, final_f1

def get_final_text(pred_text, orig_text, do_lower_case, logger, verbose_logging):
    """Project the tokenized prediction back to the original text."""
    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                            orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs
