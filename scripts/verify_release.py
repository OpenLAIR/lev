"""Fail-closed artifact and publication gates for an explicit release tree.

No path is silently ignored. Check a source snapshot without development
outputs. The default mode checks both hygiene and publication readiness.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile
import tomllib
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
CONTROL_FILES = {'MANIFEST.sha256', 'RELEASE_FILES.json'}
KINDS = {'python', 'markdown', 'json', 'toml', 'shell', 'yaml', 'svg', 'text'}
REQUIRED_READINESS = {'clean_install', 'cpu_tests', 'gpu_parity', 'weight_artifacts', 'data_rights', 'data_reconstruction',
                      'all_figure_reproduction', 'public_endpoints', 'author_review'}
FORBIDDEN_COMPONENTS = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', 'build', 'dist',
                        'node_modules', 'outputs', 'checkpoints', 'vendor', 'cache', 'caches', 'tmp'}
APPROVED_HIDDEN = {'.github', '.gitignore', '.gitleaks.toml'}
NAME = re.compile(r'[A-Za-z0-9_.@+/-]+\Z')
PRIVATE_PATH = re.compile(r'(?i)/(?:fsx/(?:home/)?|home/|users/|work/nvme/)[A-Za-z0-9_.-]+/|[A-Z]:[\\/]Users[\\/]|\\\\[A-Za-z0-9_.-]+\\[A-Za-z0-9_.-]+(?:\\|/)')
OPERATIONS = re.compile(r'(?i)\bs(?:lurm|batch|run|alloc|cancel|queue|control|info)\b|\b(?:PBS|LSB)_JOB[A-Z_]*\b|\b(?:job|allocation)[ _-]*id\s*[:=]|(?:^|[^A-Za-z])\.(?:run)time(?:/|\b)')
INTERNAL_RUN = re.compile(r'(?i)\b(?:arms-[A-Za-z0-9_-]+|h200(?:-[a-z]+)?|phase\d+-s\d+|locked|latency)-\d+(?:\.\d+)?\b|\b(?:ouro\d+|nbg\d+)-(?:ls|kev|s\d+)[A-Za-z0-9_.-]*\b')
SECRET_PATTERNS = [re.compile(p) for p in [
    r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    r'\bgh[pousr]_[A-Za-z0-9]{30,}\b',
    r'\bgithub_pat_[A-Za-z0-9_]{40,}\b',
    r'\bhf_[A-Za-z0-9]{30,}\b',
    r'\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{24,}\b',
    r'(?i)(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)\s*[=:]\s*[\"\'][A-Za-z0-9_+/=.-]{20,}[\"\']',
]]
FORBIDDEN_META = {'job', 'job_id', 'queue', 'partition_name', 'cluster', 'hostname', 'username',
                  'account', 'user', 'node', 'home_directory', 'environment_dump', 'source_run_id', 'source_snapshot_commit',
                  'source_commit', 'source_private_path', 'private_repository', 'code_sha256', 'run_dir', 'run_script'}


class GateError(ValueError):
    pass


def fail(message):
    raise GateError(message)


def safe_path(value):
    if not isinstance(value, str) or not NAME.fullmatch(value) or value.startswith('/') or value.endswith('/'):
        fail('Invalid relative artifact path')
    p = PurePosixPath(value)
    if any(x in {'.', '..', ''} for x in value.split('/')) or p.as_posix() != value:
        fail('Noncanonical artifact path')
    for part in p.parts:
        if part in FORBIDDEN_COMPONENTS or part.endswith('.egg-info') or (part.startswith('.') and part not in APPROVED_HIDDEN):
            fail('Development/cache/hidden paths cannot be approved by a manifest')
    return value


def strict_json(text):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out: fail('Duplicate JSON key')
            out[k] = v
        return out
    def constant(_): fail('Non-finite JSON number')
    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, RecursionError) as e:
        fail(f'Invalid JSON: {e}')
    check_metadata(value)
    return value


def check_metadata(value, depth=0):
    if depth > 48: fail('Metadata nesting exceeds the supported schema')
    if isinstance(value, dict):
        for k, v in value.items():
            if str(k).casefold() in FORBIDDEN_META:
                fail('Execution or private-provenance metadata is forbidden')
            # Data partition names are public scientific metadata. Resource-request
            # fields paired with a partition expose execution infrastructure.
            if str(k).casefold() == 'partition' and (v not in ('train','calibration','development','test','all') or any(str(x).casefold() in {'nodes','ntasks','gres','cpus_per_task','time_limit'} for x in value)):
                fail('Execution partition metadata is forbidden')
            check_metadata(v, depth+1)
    elif isinstance(value, list):
        for item in value: check_metadata(item, depth+1)
    elif isinstance(value, float):
        import math
        if not math.isfinite(value): fail('Non-finite metadata number')


def check_content(relative, kind, raw):
    if not isinstance(raw, bytes) or b'\x00' in raw: fail(f'{relative}: binary content is not an approved text artifact')
    try: text = raw.decode('utf-8')
    except UnicodeDecodeError: fail(f'{relative}: invalid UTF-8')
    if PRIVATE_PATH.search(text): fail(f'{relative}: machine-specific path')
    if OPERATIONS.search(text) or INTERNAL_RUN.search(text): fail(f'{relative}: execution-environment detail')
    if any(p.search(text) for p in SECRET_PATTERNS): fail(f'{relative}: credential-like content')
    suffix = PurePosixPath(relative).suffix
    expected = {'.py':'python', '.md':'markdown', '.json':'json', '.toml':'toml', '.sh':'shell',
                '.yaml':'yaml', '.yml':'yaml', '.svg':'svg', '.txt':'text'}
    if relative in {'LICENSE', 'NOTICE', '.gitignore', 'MANIFEST.sha256'}: expected_kind = 'text'
    else: expected_kind = expected.get(suffix)
    if kind not in KINDS or expected_kind != kind: fail(f'{relative}: unsupported or mismatched file type')
    if kind == 'python':
        try: ast.parse(text, filename=relative)
        except SyntaxError: fail(f'{relative}: Python syntax error')
    elif kind == 'json':
        value = strict_json(text)
        if relative == 'PROVENANCE.json':
            required = {'schema_version','artifact_id','source_repository','upstream_repository','authors',
                        'dataset_metadata','model_metadata','integrity_manifest','notice'}
            if not isinstance(value,dict) or set(value) != required or value['schema_version'] != 2:
                fail('PROVENANCE.json: unsupported public provenance schema')
            if value['source_repository'] != 'https://github.com/OpenLAIR/lev' or value['upstream_repository'] != 'https://github.com/jaredpalmer/kev':
                fail('PROVENANCE.json: repository is not an approved public identity')
        if relative == 'data/model-index.json':
            allowed = {'public_id','status','base','base_revision','seed','depths','head_temperature',
                       'per_depth_temperatures','checkpoint_sha256','artifact_id'}
            if not isinstance(value,dict) or set(value) != {'schema_version','models'} or value['schema_version'] != 1 or not isinstance(value['models'],list):
                fail('Unsupported model-index schema')
            for m in value['models']:
                if not isinstance(m,dict) or set(m) != allowed or m['public_id'] not in {'songdj/Lev-1.4B','songdj/Lev-2.6B','songdj/Lev-3B'}:
                    fail('Unsupported model identity or metadata')
                if not re.fullmatch(r'[0-9a-f]{40}', m['base_revision']) or type(m['seed']) is not int or type(m['depths']) is not int:
                    fail('Invalid model revision/depth/seed')
    elif kind == 'toml':
        try: value=tomllib.loads(text)
        except tomllib.TOMLDecodeError: fail(f'{relative}: invalid TOML')
        check_metadata(value)
        if relative == '.gitleaks.toml':
            allow = value.get('allowlists', [])
            if set(value) != {'extend','allowlists'} or value.get('extend') != {'useDefault':True} or len(allow)!=1:
                fail('Credential-scanner defaults and narrow checksum exception are required')
            rule=allow[0]
            if set(rule)!={'description','condition','paths','regexTarget','regexes'} or rule['condition']!='AND' or rule['regexTarget']!='line':
                fail('Unsupported credential-scanner exception')
            if rule['paths'] != [r'(?:^|/)data/sources\.json$'] or rule['regexes'] != [r'^\s*"test-data/level-1-api\.json": "[0-9a-f]{64}",?\s*$']:
                fail('Credential-scanner exception is broader than the reviewed checksum shape')
    elif kind == 'yaml':
        try:
            import yaml
        except ImportError: fail('PyYAML is required to validate workflow files')
        try: value = yaml.safe_load(text)
        except yaml.YAMLError: fail(f'{relative}: invalid YAML')
        if not isinstance(value, dict): fail(f'{relative}: workflow must be a mapping')
        check_metadata(value)
    elif kind == 'svg':
        if re.search(r'<!DOCTYPE|<!ENTITY', text, re.I): fail(f'{relative}: SVG declarations are forbidden')
        try: tree = ET.fromstring(text)
        except ET.ParseError: fail(f'{relative}: invalid SVG')
        if tree.tag.split('}')[-1] != 'svg': fail(f'{relative}: not an SVG root')
        for el in tree.iter():
            if el.tag.split('}')[-1].lower() in {'script','foreignobject','metadata'}: fail(f'{relative}: active or private SVG content')
            for k,v in el.attrib.items():
                key=k.split('}')[-1].lower()
                if key.startswith('on') or (key in {'href','src'} and not v.startswith('#')):
                    fail(f'{relative}: external or active SVG content')
            if el.text and re.search(r'url\(\s*[\"\']?(?!#)',el.text,re.I): fail(f'{relative}: external SVG CSS reference')
    elif kind == 'shell' and not text.startswith('#!/usr/bin/env bash\n'):
        fail(f'{relative}: unsupported shell interpreter')
    return text


def read_regular(path):
    try: descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    except OSError: fail(f'{path.name}: file cannot be opened safely')
    with os.fdopen(descriptor,'rb') as stream:
        info=os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode): fail(f'{path.name}: symlinks and special files are forbidden')
        if info.st_size > 8*1024*1024: fail(f'{path.name}: file exceeds the release limit')
        raw=stream.read(8*1024*1024+1)
        if len(raw)>8*1024*1024: fail(f'{path.name}: file exceeds the release limit')
        return raw


def inspect_tree(root):
    root = Path(root).absolute()
    if root.is_symlink() or not root.is_dir(): fail('Artifact root must be a real directory')
    inventory_raw = read_regular(root/'RELEASE_FILES.json')
    inventory = strict_json(inventory_raw.decode('utf-8'))
    if not isinstance(inventory,dict) or set(inventory) != {'schema_version','files'} or inventory['schema_version'] != 1 or not isinstance(inventory['files'],list):
        fail('Unsupported inventory schema')
    approved = {}
    for row in inventory['files']:
        if not isinstance(row,dict) or set(row) != {'path','kind'}: fail('Invalid inventory entry')
        name=safe_path(row['path'])
        if name in approved: fail('Duplicate inventory entry')
        if row['kind'] not in KINDS: fail('Invalid inventory kind')
        approved[name]=row['kind']
    if set(approved) & CONTROL_FILES: fail('Control files must not occur in the payload inventory')
    required = {'PROVENANCE.json','RELEASE_READINESS.json','.gitleaks.toml','scripts/verify_release.py'}
    if not required <= approved.keys(): fail('Required release control payload is missing')
    expected = set(approved) | CONTROL_FILES
    ancestors = {str(p) for x in expected for p in PurePosixPath(x).parents if str(p) != '.'}
    observed = set(); blobs = {}
    def walk(directory, prefix=''):
        with os.scandir(directory) as entries:
            for entry in entries:
                name = prefix+entry.name; safe_path(name)
                mode = entry.stat(follow_symlinks=False).st_mode
                if stat.S_ISLNK(mode): fail(f'{name}: symlink is forbidden')
                if stat.S_ISDIR(mode):
                    if name not in ancestors: fail(f'{name}: unapproved directory (nothing is ignored)')
                    walk(entry.path,name+'/')
                elif stat.S_ISREG(mode):
                    if name not in expected: fail(f'{name}: unapproved file')
                    observed.add(name);blobs[name]=read_regular(Path(entry.path))
                else: fail(f'{name}: special file is forbidden')
    walk(root)
    if observed != expected: fail('Missing files: '+', '.join(sorted(expected-observed)))
    if blobs['RELEASE_FILES.json'] != inventory_raw: fail('Inventory changed during validation')
    for name,raw in blobs.items():
        check_content(name, 'json' if name=='RELEASE_FILES.json' else 'text' if name=='MANIFEST.sha256' else approved[name], raw)
    manifest={}
    for line in blobs['MANIFEST.sha256'].decode('utf-8').splitlines():
        match = re.fullmatch(r'([0-9a-f]{64})  (.+)',line)
        if not match: fail('Malformed integrity-manifest line')
        digest,name=match.groups();safe_path(name)
        if name in manifest: fail('Duplicate integrity-manifest entry')
        manifest[name]=digest
    if set(manifest) != expected-{'MANIFEST.sha256'}: fail('Manifest and approved inventory disagree')
    for name,digest in manifest.items():
        if hashlib.sha256(blobs[name]).hexdigest()!=digest: fail(f'{name}: hash mismatch')
    return blobs


def scan_secrets(root, binary=None):
    root=Path(root).absolute()
    exe = binary or shutil.which('gitleaks')
    if not exe: fail('Required credential scanner is unavailable')
    try:
        version=subprocess.run([exe,'version'],capture_output=True,text=True,check=True).stdout.strip()
        if version!='8.30.1': fail('Credential scanner must be the reviewed version 8.30.1')
        with tempfile.TemporaryDirectory(prefix='lev-scan-') as tmp:
            report=Path(tmp)/'report.json'
            run=subprocess.run([exe,'dir',str(root),'--config',str(Path(root)/'.gitleaks.toml'),'--no-banner',
                                '--ignore-gitleaks-allow','--redact','--report-format','json','--report-path',str(report)],capture_output=True,text=True)
            if run.returncode!=0 or not report.is_file() or strict_json(report.read_text(encoding='utf-8'))!=[]:
                fail('Credential scan did not pass; inspect a redacted report outside the artifact')
    except (OSError,subprocess.SubprocessError): fail('Credential scanner could not complete')


def subject_digest(blobs):
    excluded=CONTROL_FILES|{'RELEASE_READINESS.json'}
    text=''.join(name+' '+hashlib.sha256(raw).hexdigest()+'\n' for name,raw in sorted(blobs.items()) if name not in excluded)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


EVIDENCE_FIELDS = {
    'clean_install': {'python_version', 'dependency_install_exit_code', 'isolated_environment'},
    'cpu_tests': {'exit_code', 'passed', 'failed', 'skipped_network_only'},
    'gpu_parity': {'models', 'parity_passed', 'training_smoke_passed', 'readout_smoke_passed'},
    'weight_artifacts': {'models', 'hashes_verified', 'metadata_clean', 'calibration_identity_verified'},
    'data_rights': {'sources_reviewed', 'unresolved_sources', 'attribution_complete'},
    'data_reconstruction': {'partition_hashes_verified', 'missing_inputs', 'clean_rebuild_passed'},
    'all_figure_reproduction': {'figure_count', 'raw_row_reproduction_passed', 'missing_figures'},
    'public_endpoints': {'code', 'models', 'datasets', 'paper', 'all_accessible'},
    'author_review': {'reviewed_by', 'approved', 'attestation'},
}
MODELS = ['songdj/Lev-1.4B','songdj/Lev-2.6B','songdj/Lev-3B']


def evidence_valid(name, raw):
    try: d = strict_json(raw.decode('utf-8'))
    except (GateError, UnicodeError): return False
    if not isinstance(d,dict) or set(d)!={'schema_version','check_id','status','results'} or d['schema_version']!=1 or d['check_id']!=name or d['status']!='passed': return False
    r=d['results']
    if not isinstance(r,dict) or set(r)!=EVIDENCE_FIELDS[name]: return False
    integer=lambda k: type(r[k]) is int
    if name=='clean_install': return bool(re.fullmatch(r'3\.(12|13)\.\d+',str(r['python_version']))) and integer('dependency_install_exit_code') and r['dependency_install_exit_code']==0 and r['isolated_environment'] is True
    if name=='cpu_tests': return integer('exit_code') and r['exit_code']==0 and integer('passed') and r['passed']>0 and integer('failed') and r['failed']==0 and r['skipped_network_only'] is True
    if name=='gpu_parity': return r['models']==MODELS and all(r[k] is True for k in ('parity_passed','training_smoke_passed','readout_smoke_passed'))
    if name=='weight_artifacts': return r['models']==MODELS and all(r[k] is True for k in ('hashes_verified','metadata_clean','calibration_identity_verified'))
    if name=='data_rights': return integer('sources_reviewed') and r['sources_reviewed']>0 and r['unresolved_sources']==[] and r['attribution_complete'] is True
    if name=='data_reconstruction': return r['partition_hashes_verified'] is True and r['missing_inputs']==[] and r['clean_rebuild_passed'] is True
    if name=='all_figure_reproduction': return integer('figure_count') and r['figure_count']==13 and r['raw_row_reproduction_passed'] is True and r['missing_figures']==[]
    if name=='public_endpoints':
        values=[r['code'],r['paper']]+(r['models'] if isinstance(r['models'],list) else [])+(r['datasets'] if isinstance(r['datasets'],list) else [])
        return r['all_accessible'] is True and r['code']=='https://github.com/OpenLAIR/lev' and isinstance(r['models'],list) and len(r['models'])==3 and isinstance(r['datasets'],list) and len(r['datasets'])==3 and all(isinstance(x,str) and x.startswith('https://') for x in values)
    if name=='author_review': return isinstance(r['reviewed_by'],str) and bool(r['reviewed_by'].strip()) and r['approved'] is True and r['attestation']=='I reviewed this exact release artifact and approve its public release.'
    return False


def readiness_errors(blobs):
    d=strict_json(blobs['RELEASE_READINESS.json'].decode('utf-8'))
    if not isinstance(d,dict) or set(d)!={'schema_version','artifact_id','distribution','checks'} or d['schema_version']!=1 or d['artifact_id']!='OpenLAIR/lev':
        fail('Invalid readiness schema')
    if d['distribution'] not in {'candidate','release'} or not isinstance(d['checks'],dict) or set(d['checks'])!=REQUIRED_READINESS:
        fail('Missing or unknown readiness checks')
    blockers=[]
    if d['distribution']!='release': blockers.append('distribution is candidate')
    digest=subject_digest(blobs)
    for name,entry in d['checks'].items():
        if not isinstance(entry,dict) or set(entry)!={'status','reason','subject_sha256','evidence'} or entry['status'] not in {'blocked','passed'} or not isinstance(entry['reason'],str):
            fail('Invalid readiness check entry')
        if entry['status']!='passed':
            if entry['subject_sha256'] is not None or entry['evidence'] is not None or not entry['reason'].strip():
                fail('Blocked checks require a reason and no stale evidence')
            blockers.append(name+': '+entry['reason']);continue
        evidence=entry['evidence']
        if entry['subject_sha256']!=digest or not isinstance(evidence,dict) or set(evidence)!={'path','sha256'}:
            blockers.append(name+': missing evidence or artifact digest mismatch');continue
        path=safe_path(evidence['path'])
        if not path.startswith('release-evidence/') or path not in blobs or hashlib.sha256(blobs[path]).hexdigest()!=evidence['sha256']:
            blockers.append(name+': evidence is absent or does not match');continue
        if not evidence_valid(name,blobs[path]): blockers.append(name+': structured evidence does not establish the required result')
    return blockers


def verify(root, mode='release', scanner=None):
    blobs=inspect_tree(root)
    # Readiness schema is always validated, even for a candidate hygiene check.
    blockers=readiness_errors(blobs)
    # Scan the same immutable byte snapshot that is packaged, not mutable source files.
    with tempfile.TemporaryDirectory(prefix='lev-verified-') as tmp:
        for name,raw in blobs.items():
            path=Path(tmp)/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        scan_secrets(tmp,scanner)
    if mode=='release' and blockers: fail('PUBLICATION BLOCKED:\n'+'\n'.join(blockers))
    return blobs,blockers


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--mode',choices=['hygiene','release'],default='release')
    ap.add_argument('--gitleaks',help='path to the reviewed credential scanner')
    a=ap.parse_args()
    try: blobs,blockers=verify(a.root,a.mode,a.gitleaks)
    except (GateError,OSError,UnicodeError) as e: raise SystemExit(str(e))
    print(f'HYGIENE PASS: {len(blobs)} exact-inventory files, types, contents, hashes and credential scan checked.')
    if blockers: print('PUBLICATION BLOCKED: '+str(len(blockers))+' unresolved conditions; hygiene is not release approval.')
    else: print('RELEASE GATES PASS: evidence matches this artifact.')


if __name__=='__main__':main()
