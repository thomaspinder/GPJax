"""Convert python files in "examples" directory to markdown files using jupytext and nbconvert.

There's only a minor inconvenience with how supporting files are handled by nbconvert,
see https://github.com/jupyter/nbconvert/issues/1164. But these will be under a private
directory `_examples` in the docs folder, so it's not a big deal.

"""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil
import subprocess

EXCLUDE = ["utils.py"]

# Notebooks that should get the consulting CTA appended. Mirrors the
# "Tutorials" section of mkdocs.yml.
CTA_NOTEBOOKS = {
    "regression",
    "classification",
    "poisson",
    "barycentres",
    "deep_kernels",
    "graph_kernels",
    "collapsed_vi",
    "uncollapsed_vi",
    "oceanmodelling",
    "heteroscedastic_inference",
    "multioutput",
    "oilmm",
    "oak",
    "numpyro_integration",
    "spatial_linear_gp",
    "temperature_forecasting",
    "climate_extrapolation",
}

CTA_HTML = """
<section class="consulting-cta">
    <p>We currently have some <strong>availability for consulting</strong> on how Gaussian processes, Bayesian modelling, and GPJax can be integrated into your team's work. If this sounds relevant to your work, <a href="https://calendly.com/hello-1761-izqw/15-minute-meeting-clone-1">book an introductory call</a>. These calls are for consulting inquiries only. For technical usage questions and free community support, please use GitHub Discussions and the documentation below.</p>
</section>
"""


def process_file(file: Path, out_file: Path | None = None, execute: bool = False):
    """Converts a python file to markdown using jupytext and nbconvert.

    Raises:
        subprocess.CalledProcessError: If the conversion fails.
    """

    out_dir = out_file.parent
    command = f"cd {out_dir.as_posix()} && "

    out_file_name = out_file.relative_to(out_dir).as_posix()

    if execute:
        command += f"jupytext --to ipynb {file} --output - "
        command += f"| jupyter nbconvert --to markdown --execute --stdin --output {out_file_name}"
    else:
        command += f"jupytext --to markdown {file} --output {out_file_name}"

    result = subprocess.run(
        command, shell=True, check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        error_msg = f"Failed to process {file.name}: {result.stderr}"
        print(error_msg)
        raise subprocess.CalledProcessError(
            result.returncode, command, output=result.stdout, stderr=result.stderr
        )

    if file.stem in CTA_NOTEBOOKS:
        with out_file.open("a", encoding="utf-8") as fd:
            fd.write(CTA_HTML)


def is_modified(file: Path, out_file: Path):
    """Check if the output file is older than the input file."""
    return out_file.exists() and out_file.stat().st_mtime < file.stat().st_mtime


def main(args):
    # project root directory
    wdir = Path(__file__).parents[2]

    # output directory
    out_dir: Path = args.outdir
    out_dir.mkdir(exist_ok=True, parents=True)

    # copy directories in "examples" to output directory
    for dir in wdir.glob("examples/*"):
        if dir.is_dir():
            (out_dir / dir.name).mkdir(exist_ok=True, parents=True)
            for file in dir.glob("*"):
                # copy, not move!
                shutil.copy(file, out_dir / dir.name / file.name)

    # list of files to be processed
    files = [f for f in wdir.glob("examples/*.py") if f.name not in EXCLUDE]

    # process only modified files
    if args.only_modified:
        files = [f for f in files if is_modified(f, out_dir / f"{f.stem}.md")]

    print(files)

    # Track failures
    failures = []

    # process files in parallel
    if args.parallel:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {}
            for file in files:
                out_file = out_dir / f"{file.stem}.md"
                future = executor.submit(
                    process_file, file, out_file=out_file, execute=args.execute
                )
                futures[future] = file

            for future in as_completed(futures):
                file = futures[future]
                try:
                    future.result()
                    print(f"Successfully processed: {file.name}")
                except Exception as e:
                    print(f"Error processing {file.name}: {e}")
                    failures.append((file, e))
    else:
        for file in files:
            out_file = out_dir / f"{file.stem}.md"
            try:
                process_file(file, out_file=out_file, execute=args.execute)
                print(f"Successfully processed: {file.name}")
            except Exception as e:
                print(f"Error processing {file.name}: {e}")
                failures.append((file, e))

    # Report failures and exit with error code if any failed
    if failures:
        print(f"\n{len(failures)} file(s) failed to process:")
        for file, error in failures:
            print(f"  - {file.name}")
        return 1  # Return non-zero exit code
    else:
        print(f"\nAll {len(files)} file(s) processed successfully!")
        return 0


if __name__ == "__main__":
    import sys

    project_root = Path(__file__).parents[2]

    parser = ArgumentParser()
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--only_modified", action="store_true")
    parser.add_argument(
        "--outdir", type=Path, default=project_root / "docs" / "_examples"
    )
    parser.add_argument("--parallel", type=bool, default=False)
    args = parser.parse_args()

    exit_code = main(args)
    sys.exit(exit_code)
