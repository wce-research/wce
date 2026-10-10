"""Build a review draft of scientific sources; never publish or copy Git history."""
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ('app/__init__.py', 'app/session.py', 'app/repl.py',
         'docs/architecture_wce.md', 'docs/hypotheses_generative_certificates.md',
         'docs/hypotheses_active_structure.md', 'docs/hypotheses_boundary_induction.md',
         'results/predictive_coverage.json', 'results/continuum_summary.json',
         'results/active_summary.json', 'results/source_type_priors.json',
         'results/boundary_stability_calibration.json', 'results/active_stability_calibration.json')
EXCLUDED_TESTS = {'test_web.py', 'test_public_source.py', 'test_robust.py', 'test_deep.py'}


def selected_files(root=ROOT):
    paths = [root / name for name in FILES if (root / name).is_file()]
    for folder in ('core', 'language', 'physics', 'logic', 'scripts', 'demos'):
        paths.extend((root / folder).glob('*.py'))
    paths.extend(p for p in (root / 'tests').glob('*.py') if p.name not in EXCLUDED_TESTS)
    for p in paths:
        if p.is_symlink():
            raise ValueError('Symlinks cannot enter the public export')
    return sorted(set(paths), key=lambda p: p.relative_to(root).as_posix())


def build(root=ROOT):
    import io
    output = root / 'dist/public-source'
    output.mkdir(parents=True, exist_ok=True)
    entries = {}
    for p in selected_files(root):
        data = p.read_bytes()
        if b'-----BEGIN OPENSSH PRIVATE KEY-----' in data or b'-----BEGIN RSA PRIVATE KEY-----' in data:
            raise ValueError('Private key detected')
        entries[p.relative_to(root).as_posix()] = data
    entries['README.md'] = b'''# WCE scientific source - review draft\n\nExperimental primitives, transformations, invariants and composition.\n\nPython 3.12: `python -m app.repl` and `python -m unittest tests.test_wce`.\nNeural experiments require PyTorch and their protocol-specific fixtures.\nThis export excludes the website, hub implementation, cloud orchestration,\nGit history and mixed cloud/science test_robust.py and test_deep.py suites.\nTwo cross-platform numerical discrepancies remain known in scientific tests.\nNo claim of superiority over LLMs or world models is established.\n\nPUBLICATION PENDING: licence and official repository must be decided before\npublication. This draft does not grant an open-source licence.\n'''
    manifest = {'status': 'review-draft-license-pending', 'files': {n: hashlib.sha256(d).hexdigest() for n,d in sorted(entries.items())}, 'excluded_mixed_tests': sorted(EXCLUDED_TESTS)}
    entries['requirements-cpu.txt'] = b'--extra-index-url https://download.pytorch.org/whl/cpu\ntorch==2.12.0+cpu\nfilelock==3.32.3\ntyping_extensions==4.16.0\nsympy==1.14.0\nnetworkx==3.6.1\njinja2==3.1.6\nfsspec==2026.7.0\nmpmath==1.3.0\nMarkupSafe==3.0.3\n'
    manifest['files']['requirements-cpu.txt'] = hashlib.sha256(entries['requirements-cpu.txt']).hexdigest()
    entries['MANIFEST.json'] = (json.dumps(manifest, indent=2)+'\n').encode()
    archive = output / 'wce-scientific-source.tar'
    with tarfile.open(archive, 'w', format=tarfile.USTAR_FORMAT) as tar:
        for name, data in sorted(entries.items()):
            info = tarfile.TarInfo('wce/'+name)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            tar.addfile(info, io.BytesIO(data))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'{len(manifest["files"])} files; review draft: {archive.relative_to(root)}')
    return archive


if __name__ == '__main__':
    build()
