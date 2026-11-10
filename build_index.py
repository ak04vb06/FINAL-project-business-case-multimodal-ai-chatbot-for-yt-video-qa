"""Builds the ChromaDB index from the videos listed in data/videos.json.

    python build_index.py            # index the videos that are not indexed yet
    python build_index.py --rebuild  # delete the index and build it again

Transcripts are cached in data/transcripts/, so YouTube is only asked once per video
(it blocks repeated requests). A rebuild reuses the cache.
"""
import argparse
import sys

from balade import config
from balade.index import add_video, chunk_count, is_indexed, load_catalog, make_embeddings, open_store
from balade.transcripts import TranscriptError, get_transcript


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="delete the collection and index everything again")
    args = parser.parse_args()

    print(f"Embeddings: {config.EMBEDDING_PROVIDER} / {config.EMBEDDING_MODEL}")
    embeddings = make_embeddings()
    store = open_store(embeddings)
    if args.rebuild:
        store.delete_collection()
        store = open_store(embeddings)

    failures = 0
    for video in load_catalog():
        if is_indexed(store, video.video_id):
            print(f"  = {video.title} (already indexed)")
            continue
        try:
            transcript = get_transcript(video.video_id, config.TRANSCRIPTS_DIR, config.TRANSCRIPT_LANGUAGES)
        except TranscriptError as error:
            print(f"  ! {video.title}: no transcript ({error.reason})")
            failures += 1
            continue
        count = add_video(store, video, transcript.segments)
        print(f"  + {video.title}: {count} chunks")

    print(f"Done: {chunk_count(store)} chunks in {config.CHROMA_DIR}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
