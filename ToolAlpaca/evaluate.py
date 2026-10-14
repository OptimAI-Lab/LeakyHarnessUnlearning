"""Run simulation, model evaluation, and judging with one command."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen
from urllib.error import URLError

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('single', 'harness'), default='single')
    parser.add_argument('--model', default='TangQiaoYu/ToolAlpaca-7B')
    parser.add_argument('--harness-model', default='TangQiaoYu/ToolAlpaca-13B')
    parser.add_argument('--split', choices=('forget', 'retain', 'general'), default='forget')
    parser.add_argument('--output', help='New output directory')
    parser.add_argument('--simulator-model', default='gpt-3.5-turbo')
    parser.add_argument('--judge-model', default='gpt-4-0613')
    args = parser.parse_args()
    if not os.environ.get('OPENAI_API_KEY'):
        parser.error('Set OPENAI_API_KEY before evaluation.')
    output = Path(args.output or f'outputs/{args.split}_{args.mode}').resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Choose a new --output directory; existing results are never overwritten.')
    data = (ROOT / 'data/eval_simulated.json' if args.split == 'general' else
            ROOT / f'tool_unlearn/data/{args.split}_eval_100.json')
    env = os.environ.copy()
    env.update(ACTOR_MODEL=args.model, PLANNER_MODEL=args.harness_model)
    env.setdefault('ACTOR_DEVICE', '0' if args.mode == 'single' else '1')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    server = f'http://127.0.0.1:{port}'
    output.mkdir(parents=True, exist_ok=True)
    condition = 'B0' if args.mode == 'single' else 'H1'
    def run(*command):
        subprocess.run([sys.executable, *map(str, command)], cwd=ROOT, env=env, check=True)
    with (output / 'simulator.log').open('w') as log:
        simulator = subprocess.Popen([
            sys.executable, 'instance_generation/simulator.py', '-api', str(data),
            '--port', str(port), '--model', args.simulator_model,
        ], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 120
            while True:
                if simulator.poll() is not None:
                    raise RuntimeError(f'Simulator failed; see {output / "simulator.log"}')
                try:
                    with urlopen(server + '/__simulator_status__', timeout=2) as response:
                        json.load(response)
                    break
                except (URLError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f'Simulator startup timed out; see {output / "simulator.log"}')
                    time.sleep(0.5)
            run('harness/run_toolalpaca.py', '--condition', condition,
                '-api', data, '-out', output, '--server_url', server)
            (output / f'{condition}.json').rename(output / 'trajectories.json')
            (output / f'{condition}_trace.json').rename(output / 'trace.json')
            run('evaluation.py', '-api', output / 'trajectories.json',
                '-out', output / 'judge.json', '--judge_model', args.judge_model)
        finally:
            if simulator.poll() is None:
                simulator.terminate()
                try:
                    simulator.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    simulator.kill()
                    simulator.wait()
    print(f'Results: {output / "judge.json"}')


if __name__ == '__main__':
    main()
