import json
from pathlib import Path
import pytest
from modebench.cli import evaluate
from modebench.prompts import make_messages


def test_saved_responses_retain_failures_and_condition_diversity_on_success():
    path = Path(__file__).resolve().parents[1] / 'examples/responses.jsonl'
    result = evaluate([json.loads(l) for l in path.read_text().splitlines()], min_defined_prompts=1)
    for cell in result['cells']:
        assert cell['k'] == 4
        assert cell['accuracy'] == .5
        assert cell['pass_at_k'] == 1
        assert cell['distinct_at_k'] == 2
        assert cell['pcmd']['d_mode'] == 1


def test_duplicate_prompt_is_rejected():
    record = {'id':'x','level':1,'domain':'countdown','answer':{'verifier':'countdown','numbers':[1,2,3],'target':6},'responses':['1+2+3']}
    with pytest.raises(ValueError, match='duplicate'):
        evaluate([record,record])


def test_prompt_does_not_leak_answer():
    messages = make_messages(1, 'countdown', {'problem':'Use 1, 2, 3 to make 6.', 'answer':'SECRET_REFERENCE'})
    assert 'SECRET_REFERENCE' not in str(messages)
