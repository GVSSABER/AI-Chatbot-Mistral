def chunk_text(text, chunk_size=500):
    # Clean text first
    text = text.replace("\n", " ").strip()

    sentences = text.split(". ")

    chunks = []
    chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()

        # remove empty sentences
        if not sentence:
            continue

        # build chunk
        if len(chunk) + len(sentence) < chunk_size:
            chunk += sentence + ". "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + ". "

    # add last chunk
    if chunk.strip():
        chunks.append(chunk.strip())

    # FINAL CLEANING (important)
    chunks = [c for c in chunks if len(c) > 10]

    return chunks