#!/usr/bin/env python3
"""Read a CPU-only simulated-producer fixture at 1Hz. Not a GPU saturation gate.
Usage: python3 metrics_endpoint_test.py /path/to/metrics_test --seconds 130
Requires prometheus_client for an independent Prometheus exposition parse.
"""
import argparse, json, socket, statistics, subprocess, time, urllib.request
from prometheus_client.parser import text_string_to_metric_families
p=argparse.ArgumentParser();p.add_argument('fixture');p.add_argument('--seconds',type=int,default=130);p.add_argument('--frontend',action='store_true');a=p.parse_args()
proc=subprocess.Popen([a.fixture,'--serve'],stdout=subprocess.PIPE,text=True)
try:
    port=int(proc.stdout.readline()); url=f'http://127.0.0.1:{port}/metrics'
    times=[]; observed=[]; start=time.monotonic()
    prefix='pico:frontend_' if a.frontend else 'pico:'
    for i in range(a.seconds):
        time.sleep(max(0,start+i-time.monotonic()))
        t=time.monotonic()
        with urllib.request.urlopen(url,timeout=1) as r:
            assert r.status==200
            text=r.read().decode()
        times.append(time.monotonic()-t)
        families=list(text_string_to_metric_families(text))
        samples={s.name: s.value for f in families for s in f.samples if s.name.startswith('pico:')}
        for f in families:
            for s in f.samples:
                assert 'model_name' in s.labels, s
        observed.append(samples)
    # A partial header, an aborted reader and a malformed request must not kill or serialize the server.
    with socket.create_connection(('127.0.0.1',port),timeout=1) as slow:
        slow.sendall(b'GET /')
        with urllib.request.urlopen(url,timeout=1) as r: assert r.status==200
    with socket.create_connection(('127.0.0.1',port),timeout=1) as c:
        c.sendall(b'POST /metrics HTTP/1.1\r\nHost: test\r\n\r\n')
        assert b'404' in c.recv(1024)
    with urllib.request.urlopen(url,timeout=1) as r: assert r.status==200
    ordered=sorted(times);p99=ordered[min(len(ordered)-1,int(.99*len(ordered)))]
    assert p99<.1 and max(times)<1
    assert max(x.get(prefix+('requests_running' if a.frontend else 'num_requests_running'),0) for x in observed)==16
    assert max(x.get(prefix+('requests_waiting' if a.frontend else 'num_requests_waiting'),0) for x in observed)==3
    assert max(x.get(prefix+'generation_tokens_total',0) for x in observed)==32
    assert observed[0][prefix+('requests_running' if a.frontend else 'num_requests_running')]==0
    heartbeat=prefix+'metrics_heartbeat_timestamp_seconds'
    assert observed[-1][heartbeat]>observed[0][heartbeat]
    if not a.frontend:
        assert observed[-1]['pico:time_to_first_token_seconds_count']==16
        assert observed[-1]['pico:request_queue_time_seconds_count']==16
        assert observed[-1]['pico:request_time_per_output_token_seconds_count']==16
        terminal=[s for f in families for s in f.samples if s.name=='pico:requests_finished_total']
        assert sum(s.value for s in terminal)==19
        assert next(s.value for s in terminal if s.labels['finished_reason']=='error')==1
        phase=[s for f in families for s in f.samples if s.name=='pico:inference_progress_timestamp_seconds' and s.labels['phase']=='decode'][0]
        assert observed[-1][heartbeat]-phase.value>120
    print(json.dumps({'endpoint':'frontend' if a.frontend else 'rank','kind':'CPU synthetic producer-stall only; NOT GPU qualification',
        'scrapes':len(times),'successes':len(times),'failures':0,'p99_ms':1000*p99,'max_ms':1000*max(times),
        'mean_ms':1000*statistics.mean(times),'producer_stall_seconds':125,
        'max_simulated_running':16,'max_simulated_waiting':3,'known_output_tokens':32},indent=2))
finally:
    proc.wait(timeout=10)
