# tests/test_preprocess.py

from preprocess import clean, detect_sarcasm_features, expand_contractions, batch_clean


def test_clean_removes_url():
    assert "https" not in clean("Visit https://example.com for more info")


def test_clean_removes_mention():
    assert "@user" not in clean("@user thanks so much!")


def test_clean_hashtag_keeps_word():
    result = clean("#awesome product")
    assert "awesome" in result


def test_clean_preserves_negation():
    result = clean("not bad at all")
    assert "not" in result


def test_clean_lowercase():
    result = clean("THIS IS GREAT")
    assert result == result.lower()


def test_expand_contractions():
    assert "will not" in expand_contractions("won't do it")
    assert "cannot"   in expand_contractions("can't stop")


def test_sarcasm_features_keys():
    feats = detect_sarcasm_features("great, just wonderful...")
    assert "has_ellipsis" in feats
    assert "all_caps_ratio" in feats


def test_sarcasm_ellipsis_detected():
    feats = detect_sarcasm_features("Oh great...")
    assert feats["has_ellipsis"] == 1.0


def test_batch_clean_length():
    texts  = ["Hello world!", "Bad product.", "Okay I guess."]
    result = batch_clean(texts)
    assert len(result) == len(texts)
