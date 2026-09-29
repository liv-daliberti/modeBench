"""Immutable pre-refactor behavior and independent mathematical witness review."""
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random

import pytest
from modebench.historical_prompts import grade_response
from modebench.metrics import mode_diversity, rarefied_distinct
from modebench.data import load_split, split_record
from tests.conformance_oracle import verify_witness, independent_mode

CORPUS = json.loads((Path(__file__).parent/'fixtures/conformance-v1.json').read_text())
FIXTURES = CORPUS['fixtures']
CASES = [(f,c) for f in FIXTURES for c in f['cases']]


def test_corpus_is_frozen_and_covers_all_25_cells():
    data=(Path(__file__).parent/'fixtures/conformance-v1.json').read_bytes()
    assert hashlib.sha256(data).hexdigest() == 'be0f2a4e8d99217a1e52b7f6570593d76c5e3de6e8646d4c7d15fff63258e6e9'
    assert CORPUS['baseline_commit']=='0ae27d301c144411d6a3b67b9a3d2483d33b02bb'
    assert {(f['level'],f['domain']) for f in FIXTURES} == {(level,domain) for level in range(1,6) for domain in ('countdown','graph_coloring','python_factors','mathir','pantry_plan')}


@pytest.mark.parametrize('fixture', FIXTURES, ids=lambda f:f['config'])
def test_witnesses_reviewed_independently_and_bound_to_real_data(fixture):
    f=fixture
    root=Path(__file__).resolve().parents[1]/'data'
    assert split_record(root,f['config'],'eval',frozen=True)['parquet_sha256']==f['parquet_sha256']
    row=load_split(root,f['config'],'eval')[f['row_index']]
    assert json.loads(row['answer'])==f['row']['answer']
    assert row['problem']==f['row']['problem']
    for witness in f['witnesses']:
        verify_witness(f['row']['answer'],f['domain'],witness)
    assert independent_mode(f['row']['answer'],f['domain'],f['witnesses'][0]) != independent_mode(f['row']['answer'],f['domain'],f['witnesses'][1])
    assert f['cases'][0]['expected']['canonical_key'] != f['cases'][1]['expected']['canonical_key']
    assert f['cases'][0]['expected']['canonical_key'] == f['cases'][2]['expected']['canonical_key']


@pytest.mark.parametrize('fixture,case', CASES, ids=[f"{f['config']}-{c['name']}" for f,c in CASES])
def test_pinned_verification_key_and_formatting_behavior(fixture,case):
    actual=grade_response(fixture['level'],fixture['domain'],fixture['row'],case['response'])
    assert {key:actual[key] for key in case['expected']}==case['expected']
    expected_status='correct' if case['expected']['verified'] else ('incorrect' if case['name']=='incorrect' else 'malformed')
    assert actual['status']==expected_status


def test_metric_properties_against_exact_pair_oracle():
    randomizer=random.Random(91821)
    for _ in range(1000):
        counts=[randomizer.randrange(20) for _ in range(randomizer.randrange(1,15))]
        k=sum(counts)
        if k<2: continue
        exact=Fraction(sum(a*b for i,a in enumerate(counts) for b in counts[i+1:]), k*(k-1)//2)
        observed=mode_diversity(counts)
        assert 0<=observed<=1
        assert observed==pytest.approx(float(exact),abs=1e-15)
        permuted=counts[:];randomizer.shuffle(permuted)
        assert mode_diversity(permuted)==observed
        assert mode_diversity({'renamed'+str(i):v for i,v in enumerate(counts)})==observed
        assert rarefied_distinct(counts,2)==pytest.approx(1+observed,abs=1e-12)


def test_countdown_commutative_canonicalization_property():
    randomizer=random.Random(441)
    for _ in range(60):
        a,b,c=[randomizer.randrange(1,30) for _ in range(3)]
        row={'answer':{'verifier':'countdown','numbers':[a,b,c],'target':a+b+c}}
        first=grade_response(1,'countdown',row,f'{a}+({b}+{c})')
        same_tree=grade_response(1,'countdown',row,f'({c}+{b})+{a}')
        assert first['status']==same_tree['status']=='correct'
        assert first['canonical_key']==same_tree['canonical_key']


@pytest.mark.parametrize('mutation', ['accept_bad','collapse_keys','reject_boxed'])
def test_conformance_gate_detects_semantic_mutations(monkeypatch,mutation):
    from ops import check_conformance
    original=check_conformance.grade_response
    def changed(level,domain,row,text):
        observed=original(level,domain,row,text)
        if mutation=='accept_bad' and not observed['verified']:
            observed.update(status='correct',verified=True,canonical_key='fabricated')
        if mutation=='collapse_keys' and observed['verified']:
            observed['canonical_key']='one-mode'
        if mutation=='reject_boxed' and text.startswith('\\boxed'):
            observed.update(status='malformed',verified=False,canonical_key=None)
        return observed
    monkeypatch.setattr(check_conformance,'grade_response',changed)
    with pytest.raises(AssertionError,match='verifier drift'):
        check_conformance.check(CORPUS)


def test_evaluation_scores_are_invariant_to_prompt_and_response_permutations():
    from copy import deepcopy
    from modebench.evaluation import evaluate
    source = json.loads((Path(__file__).resolve().parents[1]/'src/modebench/walkthrough.json').read_text())
    records = source['records']
    expected = evaluate(records, run=source['run'])
    randomizer = random.Random(72004)
    for _ in range(8):
        shuffled = deepcopy(records)
        randomizer.shuffle(shuffled)
        for record in shuffled:
            randomizer.shuffle(record['responses'])
        actual = evaluate(shuffled, run=source['run'])
        assert actual['evaluation']['status_counts'] == expected['evaluation']['status_counts']
        for before, after in zip(expected['cells'], actual['cells']):
            for field in ('level', 'domain', 'k', 'accuracy', 'pass_at_k', 'distinct_at_k', 'pcmd'):
                assert after[field] == before[field]
            assert 0 <= after['accuracy'] <= after['pass_at_k'] <= 1
            assert 0 <= after['distinct_at_k'] <= after['k']
