from balade.chunking import chunk_segments
from balade.transcripts import Segment


def segments(n, words=8):
    return [Segment(" ".join(f"w{i}_{j}" for j in range(words)), float(i * 4)) for i in range(n)]


def test_chunks_respect_size_and_keep_start_times():
    segs = segments(100)
    chunks = chunk_segments(segs, size=300, overlap=60)
    assert len(chunks) > 5
    assert all(len(c.text) <= 300 for c in chunks)
    starts = {s.text.split()[0]: s.start for s in segs}
    for chunk in chunks:
        assert chunk.start == starts[chunk.text.split()[0]]     # start = time of its first caption line


def test_consecutive_chunks_overlap_and_cover_everything():
    segs = segments(60)
    chunks = chunk_segments(segs, size=300, overlap=60)
    for previous, current in zip(chunks, chunks[1:]):
        assert current.text.split()[0] in previous.text          # the next chunk starts inside the previous one
    joined = " ".join(c.text for c in chunks)
    assert all(s.text in joined for s in segs)


def test_edge_cases():
    assert chunk_segments([]) == []
    long = [Segment("x" * 1000, 3.0)]
    assert [c.start for c in chunk_segments(long, size=800)] == [3.0]
