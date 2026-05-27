"""
Unit/integration tests for aCBL core functions.
No API calls, no external dependencies beyond numpy/sklearn.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from app import (
    is_braille_char,
    extract_braille,
    validate_braille_response,
    braille_to_dots,
    dots_to_braille,
    braille_string_to_dot_matrix,
    compute_convergence,
    compute_majority_consensus,
    braille_to_text_approx,
    pairwise_dot_similarity,
    cluster_codebooks,
    Provider,
    SYSTEM_PROMPT,
)


# ─── is_braille_char ──────────────────────────────────────────────────────────

class TestIsBrailleChar:
    def test_braille_range_start(self):
        assert is_braille_char('\u2800') is True

    def test_braille_range_end(self):
        assert is_braille_char('\u28FF') is True

    def test_braille_middle(self):
        assert is_braille_char('⠗') is True  # U+2817 = r

    def test_non_braille_ascii(self):
        assert is_braille_char('a') is False

    def test_non_braille_emoji(self):
        assert is_braille_char('🧠') is False


# ─── extract_braille ──────────────────────────────────────────────────────────

class TestExtractBraille:
    def test_pure_braille(self):
        assert extract_braille('⠗⠽⠁⠝') == '⠗⠽⠁⠝'

    def test_mixed_content(self):
        assert extract_braille('Hello ⠗⠽⠁⠝ world') == '⠗⠽⠁⠝'

    def test_no_braille(self):
        assert extract_braille('Hello world') == ''

    def test_empty_string(self):
        assert extract_braille('') == ''


# ─── validate_braille_response ─────────────────────────────────────────────────

class TestValidateBrailleResponse:
    def test_pure_braille_valid(self):
        is_valid, braille, purity = validate_braille_response('⠗⠽⠁⠝')
        assert is_valid is True
        assert braille == '⠗⠽⠁⠝'
        assert purity == 1.0

    def test_mostly_braille_valid(self):
        # 4 braille + 1 space → purity should still pass (spaces excluded)
        is_valid, braille, purity = validate_braille_response('⠗⠽ ⠁⠝')
        assert is_valid is True
        assert braille == '⠗⠽⠁⠝'

    def test_mostly_latin_invalid(self):
        is_valid, braille, purity = validate_braille_response('Hello world ⠁')
        assert is_valid is False

    def test_empty_string(self):
        is_valid, braille, purity = validate_braille_response('')
        assert is_valid is False
        assert purity == 0.0

    def test_whitespace_only(self):
        is_valid, braille, purity = validate_braille_response('   \n\t  ')
        assert is_valid is False


# ─── braille_to_dots / dots_to_braille ─────────────────────────────────────────

class TestBrailleDotConversion:
    def test_empty_cell(self):
        # U+2800 = no dots raised
        dots = braille_to_dots('\u2800')
        assert dots == [False] * 8

    def test_full_cell(self):
        # U+28FF = all 8 dots raised
        dots = braille_to_dots('\u28FF')
        assert dots == [True] * 8

    def test_roundtrip(self):
        for offset in range(256):
            ch = chr(0x2800 + offset)
            assert dots_to_braille(braille_to_dots(ch)) == ch

    def test_dot_1_only(self):
        # Dot 1 = bit 0 → offset 1 → U+2801 = ⠁ (a)
        dots = braille_to_dots('⠁')
        assert dots[0] is True
        assert all(d is False for d in dots[1:])

    def test_dots_to_braille_empty(self):
        assert dots_to_braille([False] * 8) == '\u2800'

    def test_dots_to_braille_full(self):
        assert dots_to_braille([True] * 8) == '\u28FF'


# ─── braille_string_to_dot_matrix ──────────────────────────────────────────────

class TestBrailleStringToDotMatrix:
    def test_basic(self):
        matrix = braille_string_to_dot_matrix('⠁⠃')
        assert len(matrix) == 2
        assert matrix[0][0] is True  # dot 1 of 'a'

    def test_filters_non_braille(self):
        matrix = braille_string_to_dot_matrix('⠁x⠃')
        assert len(matrix) == 2  # 'x' filtered out


# ─── compute_convergence ──────────────────────────────────────────────────────

class TestComputeConvergence:
    def test_identical_responses(self):
        conv = compute_convergence(['⠗⠽⠁⠝', '⠗⠽⠁⠝', '⠗⠽⠁⠝'])
        assert conv == 1.0

    def test_completely_different(self):
        # U+2800 (no dots) vs U+28FF (all dots) = 0% agreement
        conv = compute_convergence(['\u2800\u2800', '\u28FF\u28FF'])
        assert conv == 0.0

    def test_single_response(self):
        conv = compute_convergence(['⠗⠽⠁⠝'])
        assert conv == 0.0

    def test_empty_list(self):
        conv = compute_convergence([])
        assert conv == 0.0

    def test_partial_agreement(self):
        # Same first cell, different second cell
        conv = compute_convergence(['⠗⠁', '⠗⠃'])
        assert 0.0 < conv < 1.0


# ─── compute_majority_consensus ────────────────────────────────────────────────

class TestComputeMajorityConsensus:
    def test_identical(self):
        result = compute_majority_consensus(['⠗⠽⠁⠝', '⠗⠽⠁⠝', '⠗⠽⠁⠝'])
        assert result == '⠗⠽⠁⠝'

    def test_majority_wins(self):
        # Two say ⠁ (dot 1), one says ⠃ (dots 1,2)
        # Majority for dot 1: 3/3 → True
        # Majority for dot 2: 1/3 → False
        # Result: ⠁
        result = compute_majority_consensus(['⠁', '⠁', '⠃'])
        assert result == '⠁'

    def test_empty_input(self):
        result = compute_majority_consensus([])
        assert result == ''

    def test_truncates_to_shortest(self):
        result = compute_majority_consensus(['⠁⠃', '⠁'])
        assert len(result) == 1


# ─── braille_to_text_approx ───────────────────────────────────────────────────

class TestBrailleToTextApprox:
    def test_name_ryan(self):
        result = braille_to_text_approx('⠗⠽⠁⠝')
        assert result == 'ryan'

    def test_empty(self):
        assert braille_to_text_approx('') == ''

    def test_unknown_char(self):
        # U+28FF doesn't have a mapping → hex fallback
        result = braille_to_text_approx('\u28FF')
        assert '[0x28ff]' in result

    def test_space(self):
        result = braille_to_text_approx('⠁⠀⠃')
        assert result == 'a b'


# ─── pairwise_dot_similarity ──────────────────────────────────────────────────

class TestPairwiseDotSimilarity:
    def test_identical(self):
        assert pairwise_dot_similarity('⠗⠽⠁⠝', '⠗⠽⠁⠝') == 1.0

    def test_completely_different(self):
        assert pairwise_dot_similarity('\u2800', '\u28FF') == 0.0

    def test_empty_strings(self):
        assert pairwise_dot_similarity('', '⠁') == 0.0

    def test_symmetry(self):
        a, b = '⠗⠽⠁⠝', '⠗⠽⠃⠝'
        assert pairwise_dot_similarity(a, b) == pairwise_dot_similarity(b, a)


# ─── cluster_codebooks ─────────────────────────────────────────────────────────

class TestClusterCodebooks:
    def test_identical_outputs_single_cluster(self):
        model_braille = {
            'model_a': '⠗⠽⠁⠝',
            'model_b': '⠗⠽⠁⠝',
            'model_c': '⠗⠽⠁⠝',
        }
        clusters = cluster_codebooks(model_braille)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_two_distinct_codebooks(self):
        model_braille = {
            'model_a': '⠗⠽⠁⠝',
            'model_b': '⠗⠽⠁⠝',
            'model_c': '\u28FF\u28FF\u28FF\u28FF',  # completely different
        }
        clusters = cluster_codebooks(model_braille)
        assert len(clusters) == 2

    def test_empty_input(self):
        clusters = cluster_codebooks({})
        assert clusters == []

    def test_all_empty_braille(self):
        model_braille = {'a': '', 'b': '', 'c': ''}
        clusters = cluster_codebooks(model_braille)
        assert clusters == []

    def test_threshold_sensitivity(self):
        # Very strict threshold should separate more models
        model_braille = {
            'a': '⠗⠽⠁⠝',
            'b': '⠗⠽⠃⠝',  # slightly different
        }
        # At 0.85 they might still cluster
        clusters_loose = cluster_codebooks(model_braille, threshold=0.5)
        # At 0.99 they should separate
        clusters_strict = cluster_codebooks(model_braille, threshold=0.99)
        assert len(clusters_loose) <= len(clusters_strict)


# ─── Provider ──────────────────────────────────────────────────────────────────

class TestProvider:
    def test_provider_creation(self):
        p = Provider(
            name="Test",
            api_url="https://example.com/v1/chat/completions",
            api_key="test-key",
        )
        assert p.name == "Test"
        assert p.pool_size == 0

    def test_discover_no_key(self):
        p = Provider(
            name="Test",
            api_url="https://example.com/v1/chat/completions",
            api_key="",
        )
        result = p.discover()
        assert result == 0

    def test_model_tag_unknown(self):
        p = Provider(
            name="Test",
            api_url="https://example.com/v1/chat/completions",
            api_key="test",
        )
        tag = p.model_tag("unknown/model")
        assert tag == "unknown/model"


# ─── SYSTEM_PROMPT ─────────────────────────────────────────────────────────────

class TestSystemPrompt:
    def test_mentions_braille(self):
        assert 'braille' in SYSTEM_PROMPT.lower()

    def test_mentions_unicode_range(self):
        assert 'U+2800' in SYSTEM_PROMPT
        assert 'U+28FF' in SYSTEM_PROMPT

    def test_mentions_8_dot(self):
        assert '8-dot' in SYSTEM_PROMPT
