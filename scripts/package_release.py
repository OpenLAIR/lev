"""Create a verified source archive; unpublished-candidate packaging is explicit."""
import argparse
from pathlib import Path
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parent))
from verify_release import ROOT, GateError, verify


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--candidate',action='store_true',help='package an unpublished candidate despite readiness blockers')
    ap.add_argument('--gitleaks')
    a=ap.parse_args()
    if a.out.resolve().is_relative_to(a.root.resolve()):ap.error('Archive must be outside the verified artifact')
    if a.out.exists():ap.error('Refusing to overwrite an existing archive')
    try: blobs,blockers=verify(a.root,'hygiene' if a.candidate else 'release',a.gitleaks)
    except (GateError,OSError) as e:raise SystemExit(str(e))
    # Write exactly the bytes that passed validation, avoiding a second mutable-file read.
    a.out.parent.mkdir(parents=True,exist_ok=True)
    prefix='lev-candidate' if a.candidate else 'lev'
    with zipfile.ZipFile(a.out,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name,raw in sorted(blobs.items()):
            entry=zipfile.ZipInfo(prefix+'/'+name,date_time=(2026,1,1,0,0,0))
            entry.external_attr=(0o100755 if name.endswith('.sh') else 0o100644)<<16
            entry.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(entry,raw)
        archive.comment=b'UNPUBLISHED CANDIDATE; release readiness must pass separately' if a.candidate else b'Lev source release; required gates passed'
    print(f'Created {prefix} archive with {len(blobs)} files. Publication blockers: {len(blockers)}.')


if __name__=='__main__':main()
