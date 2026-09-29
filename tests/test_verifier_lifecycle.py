"""Fault-injection tests exercise real pipes/processes, plus worker-side failures."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from modebench.diagnostics import (MAX_REPLY_BYTES, MAX_REQUEST_BYTES, MAX_RESPONSE_CHARS,
                                  UNSCORABLE, VerifierExecutionError, result)
from modebench.worker_process import BoundedWorker
from modebench.python_modebench_process import PythonFactorVerifierProcess
from modebench.verifier_worker import guarded_handle
from modebench.verifier import grade_response
from modebench.cli import evaluate
from modebench.metrics import prompt_mode_diversity

SPEC={'verifier':'python_factor_function','python_version':'factor-v1','cases':[6,10,15]}
REQUEST={'operation':'python','candidate':'lambda n: 2 if n%2==0 else 3','spec':SPEC}
COUNTDOWN={'verifier':'countdown','numbers':[1,2,3],'target':6}


def faulty_worker(monkeypatch, worker, script):
    monkeypatch.setattr(worker,'worker_command',lambda:[sys.executable,'-I','-c',script])


@pytest.mark.parametrize('script,status', [
    ("import time; time.sleep(60)", 'timeout'),
    ("import sys,time;sys.stdin.readline();sys.stdout.write('{');sys.stdout.flush();time.sleep(60)", 'timeout'),
    ("import sys;sys.stdin.readline();print('not-json',flush=True)", 'worker_failure'),
    ("import os;os._exit(9)", 'worker_failure'),
    ("import sys;sys.stdin.readline();print('{}',flush=True)", 'worker_failure'),
    (f"import sys;sys.stdin.readline();sys.stdout.write('x'*{MAX_REPLY_BYTES+1});sys.stdout.flush()", 'worker_failure'),
])
def test_bad_worker_cannot_hang_or_silently_reject(monkeypatch, script, status):
    with BoundedWorker(timeout_seconds=0.5) as worker:
        faulty_worker(monkeypatch,worker,script)
        start=time.monotonic()
        outcome=worker.request(REQUEST)
        assert outcome['status']==status
        assert time.monotonic()-start<3
        assert worker._process is None


def test_blocked_request_write_is_also_deadline_bounded(monkeypatch):
    with BoundedWorker(timeout_seconds=0.25) as worker:
        faulty_worker(monkeypatch,worker,'import time;time.sleep(60)')
        start=time.monotonic()
        outcome=worker.request({**REQUEST,'padding':'x'*300000})
        assert outcome['status']=='timeout'
        assert time.monotonic()-start<3
        assert worker._process is None


def test_dead_worker_is_reported_then_next_request_restarts():
    with PythonFactorVerifierProcess() as worker:
        assert worker.check(REQUEST['candidate'],SPEC)['status']=='correct'
        old=worker._process
        old.kill();old.wait(timeout=2)
        assert worker.check(REQUEST['candidate'],SPEC)['status']=='worker_failure'
        assert worker._process is None
        assert worker.check(REQUEST['candidate'],SPEC)['status']=='correct'
        assert worker._process.pid!=old.pid


def test_close_reaps_process_and_closes_pipes():
    worker=PythonFactorVerifierProcess()
    assert worker.check(REQUEST['candidate'],SPEC)['status']=='correct'
    process=worker._process
    worker.close();worker.close()
    assert process.poll() is not None
    assert process.stdin.closed and process.stdout.closed
    assert worker._process is None


def test_timeout_process_is_reaped_and_next_request_recovers(monkeypatch):
    worker=BoundedWorker(timeout_seconds=0.25)
    normal=worker.worker_command
    faulty_worker(monkeypatch,worker,'import time;time.sleep(60)')
    child=worker._start()
    assert worker.request(REQUEST)['status']=='timeout'
    assert child.poll() is not None and child.stdin.closed and child.stdout.closed
    monkeypatch.setattr(worker,'worker_command',normal)
    worker.timeout_seconds=5
    try:
        assert worker.request(REQUEST)['status']=='correct'
    finally:
        worker.close()


def test_legacy_validation_raises_on_infrastructure_error(monkeypatch):
    worker=PythonFactorVerifierProcess(timeout_seconds=0.25)
    faulty_worker(monkeypatch,worker,'import os;os._exit(4)')
    with pytest.raises(VerifierExecutionError) as error:
        worker.validate(REQUEST['candidate'],SPEC)
    assert error.value.diagnostic['status']=='worker_failure'


def test_wrong_malformed_and_invalid_reference_are_distinct():
    with PythonFactorVerifierProcess() as worker:
        assert worker.check('lambda n: 1',SPEC)['status']=='incorrect'
        assert worker.check("lambda n: __import__('os')",SPEC)['status']=='malformed'
        assert worker.check('lambda n: 2',{})['status']=='invalid_reference'
    assert grade_response(1,'countdown',{'answer':{}},'1+2+3')['status']=='invalid_reference'


@pytest.mark.parametrize('error,status', [(RuntimeError('bug'),'worker_failure'),
                                         (TimeoutError('late'),'timeout'),(MemoryError(),'resource_limit')])
def test_unexpected_worker_exception_is_never_wrong_answer(monkeypatch,error,status):
    def fail(_request): raise error
    monkeypatch.setattr('modebench.verifier_worker.handle',fail)
    assert guarded_handle(REQUEST)['status']==status


def test_actual_worker_timer_interrupts_a_stall(monkeypatch):
    def stall(_request): time.sleep(2)
    monkeypatch.setattr('modebench.verifier_worker.handle',stall)
    from modebench.verifier_worker import _timeout
    previous=signal.signal(signal.SIGALRM,_timeout)
    try:
        start=time.monotonic()
        assert guarded_handle(REQUEST)['status']=='timeout'
        assert time.monotonic()-start<1.5
        assert signal.getitimer(signal.ITIMER_REAL)[0]==0
    finally:
        signal.signal(signal.SIGALRM,previous)


@pytest.mark.parametrize('domain', ['graph_coloring','countdown','python_factors','mathir','pantry_plan'])
def test_response_resource_limits_apply_to_all_domains(domain):
    corpus=json.loads((Path(__file__).parent/'fixtures/conformance-v1.json').read_text())
    f=next(f for f in corpus['fixtures'] if f['level']==1 and f['domain']==domain)
    assert grade_response(1,domain,f['row'],'x'*(MAX_RESPONSE_CHARS+1))['status']=='resource_limit'


def test_oversized_request_does_not_start_worker():
    with BoundedWorker() as worker:
        assert worker.request({'padding':'x'*MAX_REQUEST_BYTES})['status']=='resource_limit'
        assert worker._process is None


@pytest.mark.parametrize('status', sorted(UNSCORABLE))
def test_metrics_cannot_treat_verifier_failure_as_a_wrong_answer(status):
    with pytest.raises(VerifierExecutionError):
        prompt_mode_diversity([{'attempts':[result(status,detail='injected')]}])


@pytest.mark.parametrize('status', sorted(UNSCORABLE))
def test_any_verifier_failure_suppresses_entire_run_aggregates(monkeypatch,status):
    from modebench.historical_prompts import grade_response as real_grade
    def sometimes_fails(level,domain,row,text):
        return result(status,detail='injected') if text=='fault' else real_grade(level,domain,row,text)
    monkeypatch.setattr('modebench.evaluation.grade_response',sometimes_fails)
    records=[{'id':'broken','level':1,'domain':'countdown','answer':COUNTDOWN,'responses':['fault','1+2+3']},
             {'id':'healthy','level':2,'domain':'countdown','answer':COUNTDOWN,'responses':['1+2+3','1*2*3']}]
    report=evaluate(records,min_defined_prompts=1)
    assert report['evaluation']['status']=='failed'
    assert report['evaluation']['status_counts'][status]==1
    for cell in report['cells']:
        assert cell['accuracy'] is None and cell['pass_at_k'] is None and cell['distinct_at_k'] is None
        assert cell['pcmd']['d_mode'] is None and cell['pcmd']['reportable'] is False
    attempts=report['cells'][0]['prompt_results'][0]['draws'][0]['attempts']
    assert attempts[0]['status']==status and attempts[1]['status']=='correct'


def test_cli_writes_failure_receipt_and_returns_nonzero(tmp_path,monkeypatch):
    from modebench.cli import main
    records=tmp_path/'input.jsonl';output=tmp_path/'output.json'
    records.write_text(json.dumps({'id':'one','level':1,'domain':'countdown','answer':COUNTDOWN,'responses':['1+2+3']})+'\n')
    monkeypatch.setattr('modebench.evaluation.grade_response',lambda *_:result('worker_failure',detail='injected'))
    monkeypatch.setattr(sys,'argv',['modebench','evaluate',str(records),'--output',str(output)])
    with pytest.raises(SystemExit) as error: main()
    assert error.value.code==3
    report=json.loads(output.read_text())
    assert report['evaluation']['status']=='failed' and report['cells'][0]['accuracy'] is None


def test_reference_worker_failure_is_not_relabelled_invalid_data(tmp_path,monkeypatch):
    from modebench.cli import main
    records=tmp_path/'input.jsonl';output=tmp_path/'output.json'
    records.write_text(json.dumps({'id':'one','level':1,'domain':'countdown','answer':COUNTDOWN,'responses':['1+2+3']})+'\n')
    def fail(*_): raise VerifierExecutionError(result('worker_failure',detail='reference worker unavailable'))
    monkeypatch.setattr('modebench.evaluation.validate_reference',fail)
    monkeypatch.setattr(sys,'argv',['modebench','evaluate',str(records),'--output',str(output)])
    with pytest.raises(SystemExit) as error: main()
    assert error.value.code==3
    report=json.loads(output.read_text())
    assert report['failure']['status']=='worker_failure'
    assert 'line 1' in report['failure']['detail']
    assert report['cells']==[]


def test_actual_worker_address_space_limit(monkeypatch):
    source=str(Path(__file__).resolve().parents[1]/'src')
    script=f"""import sys
from modebench import verifier_worker as worker
from modebench.diagnostics import WORKER_MEMORY_BYTES
def exhaust(_request):
    return bytearray(WORKER_MEMORY_BYTES * 2)
worker.handle=exhaust
worker.main()
"""
    with BoundedWorker() as process:
        faulty_worker(monkeypatch,process,script)
        diagnostic=process.request(REQUEST)
        assert diagnostic['status']=='resource_limit'
        assert process._process is None


@pytest.mark.parametrize('domain,target', [
    ('countdown','modebench.domains.countdown.verifier._countdown_eval_and_numbers'),
    ('graph_coloring','modebench.domains.graph_coloring.verifier._verify_graph_coloring_colors'),
    ('python_factors','modebench.python_modebench.execute_python_factor_candidate'),
    ('mathir','modebench.mathir._execute_mathir_commands'),
    ('pantry_plan','modebench.pantry_plan.validate_pantry_plan'),
])
def test_domain_backend_bugs_surface_as_worker_failure(monkeypatch,domain,target):
    corpus=json.loads((Path(__file__).parent/'fixtures/conformance-v1.json').read_text())
    fixture=next(f for f in corpus['fixtures'] if f['level']==2 and f['domain']==domain)
    def broken(*args,**kwargs): raise RuntimeError('injected backend defect')
    monkeypatch.setattr(target,broken)
    request={'operation':'grade','level':2,'domain':domain,'answer':fixture['row']['answer'],
             'text':fixture['cases'][0]['response']}
    assert guarded_handle(request)['status']=='worker_failure'


def test_startup_failure_is_structured(monkeypatch):
    def fail(*args,**kwargs): raise OSError('spawn unavailable')
    monkeypatch.setattr('modebench.worker_process.subprocess.Popen',fail)
    with BoundedWorker() as worker:
        assert worker.request(REQUEST)['status']=='worker_failure'


def test_interrupted_parent_cleans_up_worker(monkeypatch):
    worker=BoundedWorker()
    process=worker._start()
    def interrupt(*_args,**_kwargs): raise KeyboardInterrupt()
    monkeypatch.setattr('modebench.worker_process.selectors.EpollSelector.select',interrupt)
    with pytest.raises(KeyboardInterrupt): worker.request(REQUEST)
    assert worker._process is None and process.poll() is not None


def test_legacy_countdown_invalid_reference_remains_fail_closed():
    from modebench.grading import validated_modebench_outcome_key
    assert validated_modebench_outcome_key('1+2+3',{'verifier':'countdown'}) is None
