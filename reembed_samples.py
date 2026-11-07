"""Re-embed Steven's exact saved sample without resampling or downloading data."""

import json
from pathlib import Path

import utils


def main():
    output = Path("samples_chunked.json")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    samples = json.loads(Path("samples.json").read_text(encoding="utf-8"))
    prompts = list(dict.fromkeys(sample["content"] for sample in samples))
    if not prompts or not all(isinstance(prompt, str) for prompt in prompts):
        raise ValueError("Expected a nonempty sample with string content fields.")
    vectors = utils.embed_text(utils.get_sbert(), prompts, show_progress=True)
    embeddings = dict(zip(prompts, vectors.tolist()))
    for sample in samples:
        sample["content_embedding"] = embeddings[sample["content"]]
        sample["text_encoding"] = utils.TEXT_ENCODING
    with output.open("x", encoding="utf-8") as file:
        json.dump(samples, file, ensure_ascii=False, indent=2)
    print(f"Saved {len(samples)} responses, {len(prompts)} unique prompts to {output}")


if __name__ == "__main__":
    main()
