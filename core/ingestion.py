import re
import fitz  # PyMuPDF
from typing import List, Tuple


# ---------------------------
# CLEAN TEXT (ADVANCED)
# ---------------------------
def clean_text(text: str) -> str:
    text = text.replace("\n", " ")

    # Fix hyphenation (important for PDFs)
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)

    # Remove weird unicode but keep basic punctuation
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# ---------------------------
# SENTENCE SPLITTING (BETTER THAN WORD SPLIT)
# ---------------------------
def split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


# ---------------------------
# SMART CHUNKING (🔥 CORE UPGRADE)
# ---------------------------
def semantic_chunks(
    text: str,
    max_words: int = 180,
    overlap_sentences: int = 2,
) -> List[str]:

    sentences = split_sentences(text)

    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        words = sent.split()
        if current_len + len(words) > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))

                # overlap
                current_chunk = current_chunk[-overlap_sentences:]
                current_len = sum(len(s.split()) for s in current_chunk)

        current_chunk.append(sent)
        current_len += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ---------------------------
# DETECT LOW-QUALITY PAGE
# ---------------------------
def is_low_quality(text: str) -> bool:
    if not text or len(text) < 50:
        return True

    # too many symbols → likely garbage
    symbol_ratio = len(re.findall(r'[^a-zA-Z0-9 ]', text)) / len(text)
    if symbol_ratio > 0.4:
        return True

    return False


# ---------------------------
# EXTRACT TEXT (MULTI STRATEGY)
# ---------------------------
def extract_page_text(page) -> str:
    text = page.get_text("text")

    # fallback for structured PDFs
    if not text or len(text.strip()) < 50:
        blocks = page.get_text("blocks")
        text = " ".join(b[4] for b in blocks if len(b) > 4)

    return text


# ---------------------------
# INGEST PDF (FINAL)
# ---------------------------
def ingest_pdf(
    file_path: str,
    doc_id: str,
    filename: str,
) -> Tuple[List[str], List[dict]]:

    doc = fitz.open(file_path)

    all_chunks = []
    all_metadata = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        raw_text = extract_page_text(page)
        cleaned = clean_text(raw_text)

        if is_low_quality(cleaned):
            continue

        # semantic chunking
        chunks = semantic_chunks(cleaned)

        for idx, chunk in enumerate(chunks):
            if len(chunk) < 80:
                continue

            chunk_id = f"{doc_id}_p{page_num+1}_c{idx}"

            all_chunks.append(chunk)

            all_metadata.append({
                "doc_id": doc_id,
                "filename": filename,
                "page": page_num + 1,
                "chunk_index": idx,
                "chunk_id": chunk_id,
                "length": len(chunk),
            })

    doc.close()

    if not all_chunks:
        raise ValueError(
            "No usable text extracted. PDF may be scanned. Consider OCR pipeline."
        )

    return all_chunks, all_metadata