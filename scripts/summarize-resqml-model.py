import pathlib
from nrresqml.summarization.summarization import summarize_resqml


resqml_files = [
    r"full/path/to/roda.epc",
    r"full/path/to/sobrarbe.epc",
]

for f in resqml_files:
    resqml_path = pathlib.Path(f)
    outdir = resqml_path.parent / "model-summaries"
    summarize_resqml(resqml_path, outdir)
