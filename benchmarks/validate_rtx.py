"""Run synthetic RTX serving checks against an explicitly selected test server."""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
from statistics import median
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:30000')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['benchmark', 'functional', 'long'], required=True)
    parser.add_argument('--expect-optimized', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    def call(path, body=None):
        headers = {'Content-Type': 'application/json'}
        if os.environ.get('SGLANG_API_KEY'):
            headers['Authorization'] = 'Bearer ' + os.environ['SGLANG_API_KEY']
        request = urllib.request.Request(args.base_url.rstrip('/') + path,
                  None if body is None else json.dumps(body).encode(), headers)
        return urllib.request.urlopen(request, timeout=600)

    with call('/get_server_info') as response:
        info = json.load(response)
    settings = info.get('server_args', info)
    fields = ['context_length', 'max_total_tokens', 'kv_cache_dtype', 'mamba_ssm_dtype',
              'speculative_adaptive', 'enable_linear_replayssm_spec', 'disable_cuda_graph',
              'speculative_accept_threshold_single', 'speculative_accept_threshold_acc',
              'speculative_num_steps', 'speculative_num_draft_tokens', 'served_model_name']
    selected = {key: settings.get(key) for key in fields}
    (args.output / 'settings.json').write_text(json.dumps(selected, indent=2))
    assert bool(selected['speculative_adaptive']) == args.expect_optimized
    assert selected['kv_cache_dtype'] == 'fp8_e4m3' and selected['mamba_ssm_dtype'] == 'float32'
    assert not selected['disable_cuda_graph']
    assert selected['speculative_accept_threshold_single'] == selected['speculative_accept_threshold_acc'] == 1
    if args.expect_optimized:
        assert selected['enable_linear_replayssm_spec']

    if args.mode == 'benchmark':
        inputs = Path(__file__).resolve().parents[1] / 'optimizations/rtx-pro-6000-20260909/evidence/inputs'
        records = {}
        for workload in ['list', 'prose', 'code', 'reasoning']:
            ids = json.loads((inputs / (workload + '.json')).read_text())
            measured = []
            for repeat in range(4):
                start, first, last, final = time.perf_counter(), None, None, {}
                with call('/generate', {'input_ids': ids, 'stream': True,
                          'sampling_params': {'temperature': 0, 'max_new_tokens': 1024, 'sampling_seed': 1234}}) as response:
                    for line in response:
                        if not line.startswith(b'data: ') or line.strip() == b'data: [DONE]':
                            continue
                        final = json.loads(line[6:])
                        if 'error' in final:
                            raise RuntimeError(final['error'])
                        if final.get('text'):
                            now = time.perf_counter()
                            first = first or now
                            last = now
                elapsed = time.perf_counter() - start
                meta = final['meta_info']
                tokens = meta['completion_tokens']
                assert tokens == 1024 and meta['finish_reason']['type'] == 'length'
                assert meta['num_retractions'] == 0
                histogram = meta['spec_correct_drafts_histogram']
                rounds = meta['spec_verify_ct']
                assert sum(histogram) == rounds
                assert sum(i * n for i, n in enumerate(histogram)) == meta['spec_num_correct_drafts']
                row = {'wall_s': elapsed, 'wall_tps': tokens / elapsed, 'tokens': tokens,
                       'ttft_s': first - start, 'stream_tps': (tokens - 1) / (last - first),
                       'verify_rounds': rounds, 'accepted_per_round': tokens / rounds,
                       'correct_drafts_histogram': histogram, 'num_retractions': meta['num_retractions'],
                       'output_hash': hashlib.sha256(final.get('text', '').encode()).hexdigest()}
                (args.output / f'{workload}-{repeat}-response.json').write_text(json.dumps(final, indent=2))
                if repeat:
                    measured.append(row)
                print(json.dumps({'workload': workload, 'warmup': repeat == 0, 'wall_tps': row['wall_tps']}), flush=True)
            records[workload] = measured
            (args.output / 'results.json').write_text(json.dumps(records, indent=2))
        summary = {name: median(row['wall_tps'] for row in values) for name, values in records.items()}
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        return

    results = []

    def chat(messages, **extra):
        request = {'model': selected['served_model_name'], 'messages': messages,
                   'temperature': 0, 'max_tokens': 256,
                   'chat_template_kwargs': {'enable_thinking': False}, **extra}
        start = time.perf_counter()
        with call('/v1/chat/completions', request) as response:
            result = json.load(response)
        return result, time.perf_counter() - start

    def save(name, passed, response, elapsed):
        results.append({'test': name, 'passed': bool(passed), 'wall_s': elapsed, 'response': response})
        (args.output / 'acceptance.json').write_text(json.dumps(results, indent=2))
        print(json.dumps({'test': name, 'passed': bool(passed), 'wall_s': elapsed}), flush=True)
        if not passed:
            raise AssertionError('Functional check failed: ' + name)

    if args.mode == 'functional':
        tools = [{'type': 'function', 'function': {'name': 'add_numbers', 'description': 'Add two integers.',
                  'parameters': {'type': 'object', 'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                                 'required': ['a', 'b']}}}]
        for a, b in [(17, 25), (-38, 107), (901, -999)]:
            messages = [{'role': 'user', 'content': f'Call add_numbers with a={a} and b={b}.'}]
            response, elapsed = chat(messages, tools=tools, tool_choice='required')
            message = response['choices'][0]['message']
            calls = message.get('tool_calls') or []
            passed = bool(calls) and calls[0]['function']['name'] == 'add_numbers' and json.loads(calls[0]['function']['arguments']) == {'a': a, 'b': b}
            save(f'tool_{a}_{b}', passed, response, elapsed)
            response, elapsed = chat(messages + [message, {'role': 'tool', 'tool_call_id': calls[0]['id'], 'content': str(a + b)},
                                     {'role': 'user', 'content': 'Return only the integer tool result.'}], tools=tools)
            save(f'tool_return_{a}_{b}', response['choices'][0]['message'].get('content', '').strip() == str(a + b), response, elapsed)
        from PIL import Image, ImageDraw
        for color, shape in [('red', 'square'), ('blue', 'circle'), ('green', 'triangle')]:
            picture = Image.new('RGB', (160, 160), 'white')
            draw = ImageDraw.Draw(picture)
            if shape == 'square':
                draw.rectangle((30, 30, 130, 130), fill=color)
            elif shape == 'circle':
                draw.ellipse((30, 30, 130, 130), fill=color)
            else:
                draw.polygon([(80, 20), (20, 140), (140, 140)], fill=color)
            buffer = io.BytesIO()
            picture.save(buffer, format='PNG')
            uri = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
            response, elapsed = chat([{'role': 'user', 'content': [{'type': 'text', 'text': 'Name the colored shape and its color.'},
                                      {'type': 'image_url', 'image_url': {'url': uri}}]}])
            text = response['choices'][0]['message'].get('content', '').lower()
            save(f'image_{color}', color in text and shape in text, response, elapsed)

    repeats = 21300 if args.mode == 'long' else 10000
    assert selected['context_length'] >= (260000 if args.mode == 'long' else 125000)
    keys = [f'START-{repeats}-8429', f'MIDDLE-{repeats}-6137', f'END-{repeats}-2951']
    filler = 'alpha beta gamma delta ' * repeats
    prompt = f'The three retrieval codes are embedded in this document.\nCode: {keys[0]}\n' + filler + f'\nCode: {keys[1]}\n' + filler + f'\nCode: {keys[2]}\n' + filler + '\nReturn all three codes exactly, in order.'
    response, elapsed = chat([{'role': 'user', 'content': prompt}])
    text = response['choices'][0]['message'].get('content', '')
    positions = [text.find(key) for key in keys]
    save(f'retrieval_{response["usage"]["prompt_tokens"]}', all(p >= 0 for p in positions) and positions == sorted(positions), response, elapsed)
    if args.mode == 'functional':
        response, elapsed = chat([{'role': 'user', 'content': 'Find the smallest positive integer x such that x mod 7 is 2, x mod 9 is 4, and x mod 11 is 6. Return a JSON object with the integer x.'}],
                                 max_tokens=2048, chat_template_kwargs={'enable_thinking': True})
        try:
            passed = json.loads(response['choices'][0]['message'].get('content', '')) == {'x': 688}
        except json.JSONDecodeError:
            passed = False
        save('reasoning_crt', passed and response['choices'][0]['finish_reason'] == 'stop', response, elapsed)


if __name__ == '__main__':
    main()
