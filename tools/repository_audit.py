"""Redacted repository audit: all reachable blobs, metadata and project files."""
import argparse
import bisect
from collections import Counter
import hashlib
import io
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import urllib.request

TOOLS = Path('/tmp/wce-repository-tools')


def bootstrap():
    TOOLS.mkdir(mode=0o700, exist_ok=True)
    version = '8.30.1'
    name = f'gitleaks_{version}_linux_x64.tar.gz'
    base = f'https://github.com/gitleaks/gitleaks/releases/download/v{version}/'
    archive = urllib.request.urlopen(base + name, timeout=60).read()
    checksums = urllib.request.urlopen(base + f'gitleaks_{version}_checksums.txt', timeout=60).read().decode()
    expected = next(line.split()[0] for line in checksums.splitlines() if line.split()[-1] == name)
    if hashlib.sha256(archive).hexdigest() != expected:
        raise RuntimeError('Official scanner archive checksum mismatch')
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        member = stream.getmember('gitleaks')
        (TOOLS / 'gitleaks').write_bytes(stream.extractfile(member).read())
    (TOOLS / 'gitleaks').chmod(0o700)
    subprocess.run(['python3', '-m', 'venv', str(TOOLS / 'venv')], check=True)
    subprocess.run([str(TOOLS / 'venv/bin/pip'), 'install', 'git-filter-repo==2.47.0'], check=True)
    print('VERIFIED official Gitleaks checksum; history tool installed outside Git')


PATTERNS = {
    'PUBLIC_IPV4': re.compile(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])'),
    'HOME_PATH': re.compile(r'/(?:home|Users)/[A-Za-z0-9_.-]+'),
    'AWS_ACCOUNT_ID': re.compile(r'(?<![\w.])\d{12}(?![\w.])'),
    'PRIVATE_KEY': re.compile(r'-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----\s+[A-Za-z0-9+/=\s]{40,}-----END (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----'),
    'AWS_ACCESS_KEY': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'GITHUB_TOKEN': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{60,255})\b'),
    'CF_IDENTIFIER': re.compile(r'(?i)(?:zone|account)[_ -]?(?:id)?[^\n]{0,30}\b[a-f0-9]{32}\b'),
    'AWS_SSO_URL': re.compile(r'https://d-[a-f0-9]+\.awsapps\.com'),
}


def classify(data, location, findings):
    if b'\0' in data[:8192]:
        return  # Archives are covered by the independent decoded/archive scanner.
    text = data.decode('utf-8', errors='replace')
    newlines = [match.start() for match in re.finditer('\n', text)]
    for category, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            if category == 'PUBLIC_IPV4':
                try:
                    address = ipaddress.ip_address(match.group())
                except ValueError:
                    continue
                if not address.is_global:
                    continue
            findings.append({'category': category, 'location': location,
                             'line': bisect.bisect_left(newlines, match.start()) + 1,
                             'value': '[REDACTED_' + category + ']'})


def scan(repo, output, strict=False, boundary='public', policy=None):
    repo = Path(repo).resolve(); output = Path(output).resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args])
    tracked = git('ls-files', '-z').decode().split('\0')
    untracked = git('ls-files', '--others', '--exclude-standard', '-z').decode().split('\0')
    ignored = git('ls-files', '--others', '--ignored', '--exclude-standard', '-z',
                  'docs', 'ops', 'tools', 'scripts', 'results', '.env', '.env.*').decode().split('\0')
    findings = []
    literals=json.loads(Path(policy).read_text()) if policy else {}
    def inspect(data, location):
        classify(data,location,findings)
        for category, values in literals.items():
            for value in values:
                if value.encode() in data:
                    findings.append({'category':category,'location':location,'line':0,'value':'[REDACTED_'+category+']'})
    snapshot = output / 'current-tree'
    if snapshot.exists():
        shutil.rmtree(snapshot)
    snapshot.mkdir(mode=0o700)
    for name in tracked + untracked + ignored:
        if '__pycache__/' in name or name.endswith('.pyc'):
            continue
        path = repo / name
        if name and path.is_file() and not path.is_symlink():
            inspect(path.read_bytes(), 'current:' + name)
            destination = snapshot / name
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            destination.chmod(0o600)
    objects = git('rev-list', '--objects', '--all').decode().splitlines()
    process = subprocess.Popen(['git', '-C', str(repo), 'cat-file', '--batch'],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    blobs = 0
    for row in objects:
        oid, _, name = row.partition(' ')
        process.stdin.write((oid + '\n').encode()); process.stdin.flush()
        header = process.stdout.readline().decode().split()
        if len(header) != 3:
            raise RuntimeError('Unexpected Git object response')
        body = process.stdout.read(int(header[2])); process.stdout.read(1)
        if header[1] == 'blob':
            blobs += 1
            inspect(body, 'history:' + name + ':' + oid)
        elif header[1] in ('commit', 'tag'):
            inspect(body, 'metadata:' + oid)
    process.stdin.close(); process.wait()
    for mode, extra in [('git', ['--log-opts=--all --full-history']), ('dir', [])]:
        result = subprocess.run([str(TOOLS / 'gitleaks'), mode, str(repo if mode == 'git' else snapshot), '--redact=100',
            '--no-banner', '--max-decode-depth=2', '--max-archive-depth=2',
            '--report-format=json', '--report-path=' + str(output / ('gitleaks-' + mode + '.json')),
            *extra], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode not in (0, 1):
            raise RuntimeError('Secret scanner failed; raw output suppressed')
        report = json.loads((output / ('gitleaks-' + mode + '.json')).read_text())
        print(json.dumps({'scanner': 'gitleaks-' + mode, 'findings': len(report),
                          'rules': dict(Counter(row['RuleID'] for row in report))}))
    report = {'reachable_commits': len(git('rev-list', '--all').splitlines()),
              'unique_blobs': blobs, 'tracked_files': len([x for x in tracked if x]),
              'untracked_project_files': len([x for x in untracked if x]),
              'ignored_project_artifacts': len([x for x in ignored if x]),
              'classifications': dict(Counter(row['category'] for row in findings)),
              'findings': findings}
    (output / 'sensitivity.json').write_text(json.dumps(report, indent=2) + '\n')
    for path in output.glob('*.json'):
        path.chmod(0o600)
    print(json.dumps({key: value for key, value in report.items() if key != 'findings'}))
    shutil.rmtree(snapshot)
    if strict:
        secret_count = sum(len(json.loads((output / ('gitleaks-' + mode + '.json')).read_text())) for mode in ('git', 'dir'))
        prohibited_categories={'PRIVATE_KEY','AWS_ACCESS_KEY','GITHUB_TOKEN'}
        if boundary=='public':
            prohibited_categories.update({'HOME_PATH','CF_IDENTIFIER','AWS_SSO_URL',*literals.keys()})
        prohibited = [row for row in findings if row['category'] in prohibited_categories]
        if secret_count or prohibited:
            raise RuntimeError('Unresolved secret/sensitivity findings; inspect redacted report')


def summary(output):
    report=json.loads((Path(output)/'sensitivity.json').read_text())
    groups={}
    for row in report['findings']:
        scope,name,*_=row['location'].split(':')
        groups.setdefault(row['category'],set()).add(scope+':'+name)
    print(json.dumps({category: sorted(paths) for category, paths in groups.items()},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['bootstrap', 'scan', 'summary'])
    parser.add_argument('--repo', default='.')
    parser.add_argument('--output', default='/tmp/wce-repository-audit')
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--boundary', choices=['public','private'],default='public')
    parser.add_argument('--policy')
    args = parser.parse_args()
    if args.action == 'bootstrap':
        bootstrap()
    elif args.action == 'scan':
        scan(args.repo, args.output, args.strict,args.boundary,args.policy)
    else:
        summary(args.output)
