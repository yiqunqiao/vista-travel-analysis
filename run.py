"""Run VISTA experiments with bundled data; save artefacts under outputs/."""
import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task",nargs="?",default="all",choices=["all","task1","task2","task3","task4"])
    args=parser.parse_args()
    for payload in (ROOT/"data").glob("*.gz.b64"):
        target=payload.with_suffix("")
        if not target.exists():
            target.write_bytes(base64.b64decode(payload.read_text()))
    report={}
    for task in ([f"task{i}" for i in range(1,5)] if args.task=="all" else [args.task]):
        output=ROOT/"outputs"/task;output.mkdir(parents=True,exist_ok=True)
        print(f"Running {task}...",flush=True)
        start=time.monotonic()
        env={**os.environ,"MPLBACKEND":"Agg","OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}
        result=subprocess.run([sys.executable,str(ROOT/task/"main.py")],cwd=output,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (output/"run.log").write_text(result.stdout)
        print(result.stdout[-2000:],flush=True)
        report[task]={"status":"passed" if result.returncode==0 else "failed","seconds":round(time.monotonic()-start,2),"artifacts":sorted(p.name for p in output.iterdir())}
        (ROOT/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
        if result.returncode:
            raise SystemExit(result.returncode)
        print(f"Passed {task}",flush=True)


if __name__=="__main__":
    main()
