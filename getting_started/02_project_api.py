"""getting_started/02_project_api.py

Programmatically create a Project, define Splats, add a Song with chunks,
and assign labels - no GUI required.

The project is saved to ``demo_project.splatject`` in the current directory.
"""

import os
from songsplat.core.models import Project, Song, Splat, Chunk
from songsplat.core.project_io import save_project, load_project, new_project


def main() -> None:
    proj = new_project("my demo project")
    print(f"Project: {proj.name}  (id={proj.id[:8]}...)")

    energy = Splat(name="energy",    low_label="calm",   high_label="intense", color="#FF6B6B", order=0)
    bright = Splat(name="brightness", low_label="dark",   high_label="bright",  color="#4DABF7", order=1)
    proj.splats = [energy, bright]
    print(f"Splats: {[s.name for s in proj.splats]}")

    song = Song(
        path=os.path.abspath(__file__),
        name="demo song",
        sample_rate=22050,
        duration=10.0,
    )
    for i in range(5):
        ck = Chunk(index=i, start=i * 2.0, end=(i + 1) * 2.0)
        ck.labels[energy.id] = round(i / 4, 2)
        ck.labels[bright.id] = round(1 - i / 4, 2)
        song.chunks.append(ck)

    proj.songs.append(song)
    print(f"Song '{song.name}': {len(song.chunks)} chunks")
    for ck in song.chunks:
        print(f"  chunk {ck.index}  energy={ck.labels[energy.id]:.2f}  "
              f"brightness={ck.labels[bright.id]:.2f}")

    out = "demo_project.splatject"
    save_project(proj, out)
    print(f"\nSaved to {out}")

    loaded = load_project(out)
    assert len(loaded.songs) == 1
    assert len(loaded.splats) == 2
    print(f"Reloaded: {loaded.name}  songs={len(loaded.songs)}  splats={len(loaded.splats)}")


if __name__ == "__main__":
    main()
