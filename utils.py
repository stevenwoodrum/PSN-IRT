from models.text_chunking import TEXT_ENCODING, embed_text, make_text_encoder

sbert = None

def get_sbert():
    """Return a singleton instance of an SBERT model"""
    global sbert
    if sbert is None:
        sbert = make_text_encoder(
            TEXT_ENCODING["model"],
            max_length=TEXT_ENCODING["text_max_length"],
            chunk_size=TEXT_ENCODING["text_chunk_size"],
        )
    return sbert


def create_submission_zip(zip_name: str="submission", files: list=None):
    """
    Create a ZIP archive containing the given files and directories while
    preserving their paths relative to the current working directory.

    Parameters
    ----------
    zip_name : str
        Output ZIP name *without* the .zip extension.
    files : list[str] | None
        File and directory paths to include. Directories are added recursively.
        If omitted, includes the prediction entry point, its model definition,
        and all all-data model artifacts.

    Notes
    -----
    - Archive paths retain the supplied directory structure.
    - Paths must resolve inside the current working directory.
    - Raises FileNotFoundError if any provided path does not exist.
    """
    from pathlib import Path
    from zipfile import ZIP_DEFLATED, ZipFile

    if not files:
        files = [
            "model.py",
            "utils.py",
            "models.txt",
            "requirements.txt",
            "models/Semantic_IRT.py",
            "models/text_chunking.py",
            "results/chunked/all_data",
        ]

    root = Path.cwd().resolve()
    zip_path = Path(zip_name)
    if zip_path.suffix.lower() != ".zip":
        zip_path = Path(f"{zip_path}.zip")
    if not zip_path.is_absolute():
        zip_path = root / zip_path

    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for file in files:
            path = Path(file).resolve()
            if not path.exists():
                raise FileNotFoundError(f"Required path not found: {file}")
            try:
                path.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    f"Path must be inside the current working directory: {file}"
                ) from error

            paths = [path] if path.is_file() else (
                child for child in path.rglob("*") if child.is_file()
            )
            for entry in paths:
                archive.write(entry, entry.relative_to(root))
