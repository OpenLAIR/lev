"""Adversarial source-artifact checks using synthetic data and external copies."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import pytest
from scripts import verify_release as V

SOURCE = Path(__file__).resolve().parents[1]


def seal(root):
    kinds = {'.py':'python','.md':'markdown','.json':'json','.toml':'toml','.sh':'shell','.yml':'yaml','.yaml':'yaml','.svg':'svg','.txt':'text'}
    rows=[]
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.name not in V.CONTROL_FILES:
            name=p.relative_to(root).as_posix()
            rows.append({'path':name,'kind':kinds.get(p.suffix,'text')})
    (root/'RELEASE_FILES.json').write_text(json.dumps({'schema_version':1,'files':rows}),encoding='utf-8')
    names=[r['path'] for r in rows]+['RELEASE_FILES.json']
    (root/'MANIFEST.sha256').write_text(''.join(hashlib.sha256((root/n).read_bytes()).hexdigest()+'  '+n+'\n' for n in sorted(names)),encoding='utf-8')


@pytest.fixture
def artifact(tmp_path):
    root=tmp_path/'artifact'
    root.mkdir()
    for name in ['PROVENANCE.json','RELEASE_READINESS.json','.gitleaks.toml','scripts/verify_release.py','scripts/package_release.py']:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(SOURCE/name,p)
    ready=root/'RELEASE_READINESS.json'
    d=json.loads(ready.read_text())
    for entry in d['checks'].values(): entry.update(status='blocked',reason='Synthetic pending check',subject_sha256=None,evidence=None)
    ready.write_text(json.dumps(d),encoding='utf-8')
    (root/'README.md').write_text('Synthetic source candidate\n',encoding='utf-8')
    seal(root)
    return root


def test_hygiene_and_release_are_distinct(artifact,monkeypatch):
    monkeypatch.setattr(V,'scan_secrets',lambda *args:None)
    blobs,blockers=V.verify(artifact,'hygiene')
    assert len(blockers)==len(V.REQUIRED_READINESS)+1
    with pytest.raises(V.GateError,match='PUBLICATION BLOCKED'): V.verify(artifact)
    assert V.subject_digest(blobs)


@pytest.mark.parametrize('component',['.git','.venv','__pycache__','.pytest_cache','build','dist','outputs','checkpoints','vendor','cache','unused'])
def test_unlisted_empty_directories_are_never_ignored(artifact,component):
    (artifact/component).mkdir()
    with pytest.raises(V.GateError):V.inspect_tree(artifact)


@pytest.mark.parametrize('component',['.git','__pycache__','build','outputs','sample.egg-info'])
def test_declaring_forbidden_directory_cannot_bypass_gate(artifact,component):
    (artifact/component).mkdir();(artifact/component/'item.txt').write_text('x',encoding='utf-8');seal(artifact)
    with pytest.raises(V.GateError):V.inspect_tree(artifact)


def test_symlink_rejected(artifact):
    (artifact/'alias.md').symlink_to('README.md')
    with pytest.raises(V.GateError,match='symlink'):V.inspect_tree(artifact)


def test_special_file_rejected(artifact):
    os.mkfifo(artifact/'stream.txt')
    with pytest.raises(V.GateError,match='special'):V.inspect_tree(artifact)


@pytest.mark.parametrize('name',['../outside.md','a/../b.md','/absolute.md','a//b.md','a/./b.md','a\\b.md'])
def test_noncanonical_manifest_path_rejected(artifact,name):
    p=artifact/'MANIFEST.sha256';p.write_text(p.read_text()+('0'*64)+'  '+name+'\n',encoding='utf-8')
    with pytest.raises(V.GateError):V.inspect_tree(artifact)


@pytest.mark.parametrize('change',['duplicate','missing','extra','digest'])
def test_manifest_integrity(artifact,change):
    p=artifact/'MANIFEST.sha256';lines=p.read_text().splitlines(True)
    if change=='duplicate':lines.append(lines[0])
    if change=='missing':lines.pop()
    if change=='extra':lines.append('0'*64+'  extra.md\n')
    if change=='digest':lines[0]='0'*64+lines[0][64:]
    p.write_text(''.join(lines),encoding='utf-8')
    with pytest.raises(V.GateError):V.inspect_tree(artifact)


@pytest.mark.parametrize('raw',[b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":1e999}',b'[] trailing'])
def test_malformed_json_rejected_even_when_hashed(artifact,raw):
    (artifact/'sample.json').write_bytes(raw);seal(artifact)
    with pytest.raises(V.GateError):V.inspect_tree(artifact)


@pytest.mark.parametrize('name,raw',[('bad.py',b'def broken('),('bad.toml',b'[open'),('bad.yml',b'[open'),('bad.md',b'\xff'),('bad.txt',b'a\x00b'),('bad.exe',b'text')])
def test_file_type_and_syntax_rejected(artifact,name,raw):
    (artifact/name).write_bytes(raw);seal(artifact)
    with pytest.raises(V.GateError):V.inspect_tree(artifact)


def test_inventory_type_mismatch_rejected(artifact):
    p=artifact/'RELEASE_FILES.json';d=json.loads(p.read_text());d['files'][0]['kind']='python';p.write_text(json.dumps(d),encoding='utf-8')
    with pytest.raises(V.GateError):V.inspect_tree(artifact)


@pytest.mark.parametrize('text',[
    '/'+'home'+'/fictional-user/project/file',
    's'+'batch job.sh',
    'S'+'LURM step record',
    'phase'+'7-s9-9999',
    'gh'+'p_'+'Z'*40,
    'h'+'f_'+'A'*40,
    'api'+'_key = "'+'B'*30+'"',
])
def test_private_paths_execution_and_credentials_rejected(artifact,text):
    (artifact/'extra.md').write_text(text,encoding='utf-8');seal(artifact)
    with pytest.raises(V.GateError):V.inspect_tree(artifact)


def test_execution_metadata_rejected(artifact):
    (artifact/'extra.json').write_text(json.dumps({'job'+'_id':1234}),encoding='utf-8');seal(artifact)
    with pytest.raises(V.GateError):V.inspect_tree(artifact)


def test_public_provenance_cannot_point_to_another_repository(artifact):
    p=artifact/'PROVENANCE.json';d=json.loads(p.read_text());d['source_repository']='https://example.invalid/project';p.write_text(json.dumps(d),encoding='utf-8');seal(artifact)
    with pytest.raises(V.GateError):V.inspect_tree(artifact)


def test_broadened_scanner_exception_rejected(artifact):
    p=artifact/'.gitleaks.toml';p.write_text(p.read_text().replace('condition = "AND"','condition = "OR"'),encoding='utf-8');seal(artifact)
    with pytest.raises(V.GateError):V.inspect_tree(artifact)


def test_scanner_unavailable_is_failure(artifact,monkeypatch):
    monkeypatch.setattr(V.shutil,'which',lambda _:None)
    with pytest.raises(V.GateError,match='unavailable'):V.scan_secrets(artifact)


@pytest.mark.parametrize('case',['version','exit','report'])
def test_scanner_must_complete_with_reviewed_version_and_empty_report(artifact,monkeypatch,case):
    def run(args,**kwargs):
        if args[1]=='version':return SimpleNamespace(stdout='1.0' if case=='version' else '8.30.1',returncode=0)
        report=Path(args[args.index('--report-path')+1]);report.write_text('[{"finding":true}]' if case=='report' else '[]',encoding='utf-8')
        return SimpleNamespace(stdout='',stderr='',returncode=1 if case=='exit' else 0)
    monkeypatch.setattr(V.subprocess,'run',run)
    with pytest.raises(V.GateError):V.scan_secrets(artifact,'synthetic-scanner')


VALID_RESULTS = {
    'clean_install': {'python_version':'3.13.1','dependency_install_exit_code':0,'isolated_environment':True},
    'cpu_tests': {'exit_code':0,'passed':99,'failed':0,'skipped_network_only':True},
    'gpu_parity': {'models':V.MODELS,'parity_passed':True,'training_smoke_passed':True,'readout_smoke_passed':True},
    'weight_artifacts': {'models':V.MODELS,'hashes_verified':True,'metadata_clean':True,'calibration_identity_verified':True},
    'data_rights': {'sources_reviewed':10,'unresolved_sources':[],'attribution_complete':True},
    'data_reconstruction': {'partition_hashes_verified':True,'missing_inputs':[],'clean_rebuild_passed':True},
    'all_figure_reproduction': {'figure_count':13,'raw_row_reproduction_passed':True,'missing_figures':[]},
    'public_endpoints': {'code':'https://github.com/OpenLAIR/lev','models':['https://example.invalid/model']*3,'datasets':['https://example.invalid/data']*3,'paper':'https://example.invalid/paper','all_accessible':True},
    'author_review': {'reviewed_by':'Example Reviewer','approved':True,'attestation':'I reviewed this exact release artifact and approve its public release.'},
}


def test_evidence_must_bind_current_payload_and_hash(artifact):
    d=json.loads((artifact/'RELEASE_READINESS.json').read_text());d['distribution']='release'
    (artifact/'release-evidence').mkdir()
    for name,results in VALID_RESULTS.items():
        evidence=artifact/f'release-evidence/{name}.json'
        evidence.write_text(json.dumps({'schema_version':1,'check_id':name,'status':'passed','results':results}),encoding='utf-8')
    seal(artifact);subject=V.subject_digest(V.inspect_tree(artifact))
    for name,entry in d['checks'].items():
        evidence=artifact/f'release-evidence/{name}.json'
        entry.update(status='passed',reason='synthetic test fixture only',subject_sha256=subject,evidence={'path':evidence.relative_to(artifact).as_posix(),'sha256':hashlib.sha256(evidence.read_bytes()).hexdigest()})
    p=artifact/'RELEASE_READINESS.json';p.write_text(json.dumps(d),encoding='utf-8');seal(artifact)
    assert not V.readiness_errors(V.inspect_tree(artifact))
    (artifact/'README.md').write_text('Changed payload',encoding='utf-8');seal(artifact)
    assert len(V.readiness_errors(V.inspect_tree(artifact)))==len(V.REQUIRED_READINESS)
    evidence.write_text('{}',encoding='utf-8');seal(artifact)
    assert V.readiness_errors(V.inspect_tree(artifact))


@pytest.mark.parametrize('name',sorted(V.REQUIRED_READINESS))
def test_skipped_missing_and_wrong_check_evidence_never_pass(name):
    assert not V.evidence_valid(name,b'NOT RUN. No validation performed.')
    for status in ['skipped','unknown','failed']:
        raw=json.dumps({'schema_version':1,'check_id':name,'status':status,'results':VALID_RESULTS[name]}).encode()
        assert not V.evidence_valid(name,raw)
    assert not V.evidence_valid(name,json.dumps({'schema_version':1,'check_id':name,'status':'passed','results':{}}).encode())
    assert not V.evidence_valid(name,json.dumps({'schema_version':1,'check_id':'unknown','status':'passed','results':VALID_RESULTS[name]}).encode())


@pytest.mark.parametrize('kind',['json','toml','yaml'])
@pytest.mark.parametrize('key',['hostname','username','account','user','node'])
def test_nested_private_metadata_all_formats(kind,key):
    if kind=='json':raw=json.dumps({'nested':{key:'fictional-example'}})
    elif kind=='toml':raw='[nested]\n'+key+' = "fictional-example"\n'
    else:raw='nested:\n  '+key+': fictional-example\n'
    with pytest.raises(V.GateError):V.check_content('sample.'+kind,kind,raw.encode())


def test_scanner_reads_validated_snapshot(artifact,monkeypatch):
    expected=(artifact/'README.md').read_bytes()
    def scan(root,*args):
        assert Path(root)!=artifact
        (artifact/'README.md').write_text('concurrent change',encoding='utf-8')
        assert (Path(root)/'README.md').read_bytes()==expected
    monkeypatch.setattr(V,'scan_secrets',scan)
    blobs,_=V.verify(artifact,'hygiene')
    assert blobs['README.md']==expected


def test_default_packager_refuses_candidate_even_with_fake_successful_scanner(artifact,tmp_path):
    scanner=tmp_path/'scanner'
    scanner.write_text('#!'+sys.executable+'\nimport sys,json\nfrom pathlib import Path\nif sys.argv[1]=="version": print("8.30.1")\nelse: Path(sys.argv[sys.argv.index("--report-path")+1]).write_text("[]")\n',encoding='utf-8');scanner.chmod(0o755)
    out=tmp_path/'source.zip'
    result=subprocess.run([sys.executable,str(artifact/'scripts/package_release.py'),'--root',str(artifact),'--out',str(out),'--gitleaks',str(scanner)],capture_output=True,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    assert result.returncode!=0 and 'PUBLICATION BLOCKED' in result.stderr and not out.exists()


def test_inline_secret_ignore_comments_are_disabled(artifact,monkeypatch):
    calls=[]
    def run(args,**kwargs):
        calls.append(args)
        if args[1]=='version':return SimpleNamespace(stdout='8.30.1',returncode=0)
        Path(args[args.index('--report-path')+1]).write_text('[]',encoding='utf-8')
        return SimpleNamespace(stdout='',stderr='',returncode=0)
    monkeypatch.setattr(V.subprocess,'run',run)
    V.scan_secrets(artifact,'synthetic-scanner')
    assert '--ignore-gitleaks-allow' in calls[-1]


@pytest.mark.parametrize('value',['compute-pool',123,{'nested':'resource'}])
def test_partition_requires_scientific_split_semantics(value):
    with pytest.raises(V.GateError):V.strict_json(json.dumps({'partition':value}))
    assert V.strict_json('{"partition":"development"}')['partition']=='development'
