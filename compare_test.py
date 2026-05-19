"""
compare_test.py — 3-language synthesis comparison
Run: python compare_test.py
Server must be running at http://localhost:8000
"""
import sys, os, time, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests

os.makedirs('output_compare', exist_ok=True)

TESTS = [
    ('Chinese', '早上好！希望你有美好的一天。'),
    ('English', 'Good morning! Have a wonderful day.'),
    ('Farsi',   'سلام! امیدوارم روز خوبی داشته باشید.'),
]

BASE = 'http://localhost:8000/v1/audio/speech'

results = []
print('Running 3-language comparison...\n')
for lang, text in TESTS:
    t0 = time.time()
    try:
        r = requests.post(BASE, json={
            'model': 'tts-1',
            'input': text,
            'voice': 'default',
            'response_format': 'mp3',
        }, timeout=180)
        elapsed = time.time() - t0
        size = len(r.content)
        fname = lang.lower() + '.mp3'
        path = os.path.join('output_compare', fname)
        with open(path, 'wb') as f:
            f.write(r.content)
        results.append((lang, text, size, elapsed, 'OK', path))
        print(f'  {lang}: {size:,} bytes  ({elapsed:.1f}s)  saved -> {path}')
    except Exception as e:
        elapsed = time.time() - t0
        results.append((lang, text, 0, elapsed, f'ERROR: {e}', ''))
        print(f'  {lang}: FAILED — {e}')

print()
print('='*65)
print('  LANGUAGE COMPARISON RESULTS')
print('='*65)
print(f"  {'Language':<10} {'Size':>10}  {'Time':>7}  {'Status'}")
print('-'*65)
for lang, text, size, elapsed, status, path in results:
    kb = f'{size/1024:.1f} KB' if size else '-'
    print(f"  {lang:<10} {kb:>10}  {elapsed:>5.1f}s  {status}")
print('='*65)
print()
print('Audio files saved to: output_compare/')
for lang, text, size, elapsed, status, path in results:
    if path:
        print(f'  {path}')
