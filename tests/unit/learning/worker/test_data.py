"""The tokenization boundary: a prompt is never shortened to fit a sequence budget."""
import pytest

from learning.worker.data import encode_pair_ids
from learning.worker.stages import SEQUENCE_LENGTH_CEILING, sequence_budget


def test_a_pair_that_fits_is_the_prompt_and_answer_untouched():
    feature = encode_pair_ids([1, 2, 3, 4], [5, 6], max_length=6)
    assert feature == {"input_ids": [1, 2, 3, 4, 5, 6], "attention_mask": [1] * 6,
                       "labels": [-100, -100, -100, -100, 5, 6]}


def test_a_pair_that_does_not_fit_is_refused_instead_of_cutting_the_prompt():
    assert encode_pair_ids([1, 2, 3, 4], [5, 6], max_length=5) is None


def test_an_empty_answer_is_still_an_input_error():
    with pytest.raises(ValueError):
        encode_pair_ids([1, 2], [], max_length=8)


def test_the_budget_is_a_floor_raised_to_fit_the_longest_row():
    assert sequence_budget(64, [10, 30]) == 64
    assert sequence_budget(64, [10, 300]) == 300
    assert sequence_budget(1024, []) == 1024


def test_the_budget_is_capped_at_the_ceiling_so_long_rows_are_dropped_and_counted():
    assert sequence_budget(64, [SEQUENCE_LENGTH_CEILING + 1]) == SEQUENCE_LENGTH_CEILING
    assert sequence_budget(SEQUENCE_LENGTH_CEILING + 1000, [10]) == SEQUENCE_LENGTH_CEILING
