import tempfile

from rsCNN.experiments import histories


_MOCK_HISTORY_1 = {'a': [1, 2], 'b': [5, 6]}
_MOCK_HISTORY_2 = {'a': [3, 4], 'c': [7, 8]}
_MOCK_HISTORY_COMBINED = {'a': [1, 2, 3, 4], 'b': [5, 6], 'c': [7, 8]}


def test_save_and_load_history_recovers_mock():
    file_ = tempfile.NamedTemporaryFile(mode='wb')
    histories.save_history(_MOCK_HISTORY_1, file_.name)
    history = histories.load_history(file_.name)
    file_.close()
    assert history == _MOCK_HISTORY_1


def test_combine_histories_recovers_expected():
    combined = histories.combine_histories(_MOCK_HISTORY_1, _MOCK_HISTORY_2)
    assert combined == _MOCK_HISTORY_COMBINED
