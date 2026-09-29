"""Current, versioned ModeBench interfaces for NEW training and inference.

The original frontier/E122/E123/E124 contracts remain the reproducibility
interfaces of their frozen cohorts. The bundled frozen datasets use the historical condition by default.
The neutral Python interface requires an explicit condition and its own dataset.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from . import historical_prompts as historical
from modebench.templates import TEMPLATE_FACTORY,CHAT_SURFACES

CURRENT='python_level3_neutral_v1'
HISTORICAL='registered_hints_v1'
CONDITIONS=(CURRENT,HISTORICAL)
DEFAULT_CONDITION=HISTORICAL


def profile_metadata(level,domain,condition=DEFAULT_CONDITION):
    if condition not in CONDITIONS:raise ValueError('unknown prompt condition: '+str(condition))
    level,domain=historical._identity(level,domain)
    p=historical.profile_metadata(level,domain)
    changed=level==3 and domain=='python_factors' and condition==CURRENT
    if changed:p={**p,'template_name':'qwen_level3_python_factors_neutral_v1'}
    return {**p,'schema':'modebench-current-prompt-contract-v2','prompt_condition':condition,
            'system_wording':'neutral' if changed else 'registered_original',
            'choice_informed_by_existing_evaluation':changed,
            'original_difficulty_calibration_applies_to_prompt':not changed}


def make_messages(level,domain,row,condition=DEFAULT_CONDITION):
    p=profile_metadata(level,domain,condition)
    # The historical adapter retains all task/role-marker validation.
    original=historical.make_messages(level,domain,row)
    if p['system_wording']!='neutral':return original
    rendered=TEMPLATE_FACTORY[p['template_name']](row['problem'])
    sm,um,am=CHAT_SURFACES['qwen'];assert rendered.startswith(sm) and rendered.endswith(am)
    system,user=rendered[len(sm):-len(am)].split(um)
    assert user==original[1]['content']
    return [{'role':'system','content':system},{'role':'user','content':user}]


def training_environment(level,domain,environment,condition=DEFAULT_CONDITION):
    """Apply the same registered wording to training and its evaluation loader."""
    p=profile_metadata(level,domain,condition);result=deepcopy(environment)
    original=historical.profile_metadata(level,domain)['template_name']
    if environment.get('OAT_ZERO_PROMPT_TEMPLATE') not in (original,p['template_name']):
        raise ValueError('unexpected source template; require a native registered interface')
    result['OAT_ZERO_PROMPT_TEMPLATE']=p['template_name']
    result['OAT_ZERO_MODEBENCH_PROMPT_CONDITION']=condition
    result['OAT_ZERO_MODEBENCH_LEVEL']=str(int(str(level).removeprefix('level')))
    return result


def prompt_sha256(level,domain,row,condition=DEFAULT_CONDITION):
    import json
    return hashlib.sha256(json.dumps(make_messages(level,domain,row,condition),sort_keys=True,separators=(',',':')).encode()).hexdigest()
