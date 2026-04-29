"""songsplat - command line interface

Usage:
  songsplat chunk <audio_file> [options]
  songsplat splatter <chunk_file> [options]
  songsplat train <splatdata_file(s)...> [options]
  songsplat reinforce <splat_file> <chunk_file(s)...>
  songsplat test <splat_file> <chunk_or_audio_file>
  songsplat gui
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="songsplat",
        description="songsplat - audio annotation and splat prediction toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="command")

    p_chunk = sub.add_parser("chunk", help="Split audio into chunks -> .splatchunk")
    p_chunk.add_argument("audio", help="Path to audio file (.mp3, .wav, .flac, ...)")
    p_chunk.add_argument("-o", "--output", default="", help="Output .splatchunk path")
    p_chunk.add_argument(
        "--mode", choices=["fixed", "beat"], default="fixed",
        help="Chunking mode (default: fixed)")
    p_chunk.add_argument(
        "--duration", type=float, default=2.0,
        help="Chunk duration in seconds for 'fixed' mode (default: 2.0)")
    p_chunk.add_argument(
        "--beats", type=int, default=4,
        help="Beats per chunk for 'beat' mode (default: 4)")

    p_spl = sub.add_parser("splatter", help="Label chunk values interactively -> .splatdata")
    p_spl.add_argument("chunk_file", help="Path to .splatchunk file")
    p_spl.add_argument("-o", "--output", default="", help="Output .splatdata path")
    p_spl.add_argument(
        "--data", default="", dest="existing_data",
        help="Existing .splatdata to resume editing")

    p_train = sub.add_parser("train", help="Train a model from labeled data -> .splat")
    p_train.add_argument("splatdata", nargs="+", help="One or more .splatdata files")
    p_train.add_argument("-o", "--output", default="", help="Output .splat path")
    p_train.add_argument(
        "--arch", choices=["pretrained", "raw_transformer"], default="pretrained",
        help="Model architecture (default: pretrained)")
    p_train.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30)")
    p_train.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    p_train.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")

    p_reinf = sub.add_parser(
        "reinforce", help="Review and correct model predictions -> updated .splatdata")
    p_reinf.add_argument("splat_file", help="Path to .splat model file")
    p_reinf.add_argument("chunk_files", nargs="+", help="One or more .splatchunk files")

    p_test = sub.add_parser("test", help="Run a model on audio/chunks and print predictions")
    p_test.add_argument("splat_file", help="Path to .splat model file")
    p_test.add_argument("input", help="Path to .splatchunk or raw audio file")

    sub.add_parser("gui", help="Launch the graphical app")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "chunk":
        from songsplat.cli.chunker import run_chunker
        try:
            run_chunker(
                audio_path=args.audio,
                output=args.output,
                mode=args.mode,
                duration=args.duration,
                beats_per_chunk=args.beats,
            )
        except Exception as e:
            _die(str(e))

    elif args.command == "splatter":
        from songsplat.cli.splatter import run_splatter
        try:
            run_splatter(
                chunk_path=args.chunk_file,
                output=args.output,
                existing_data=args.existing_data,
            )
        except Exception as e:
            _die(str(e))

    elif args.command == "train":
        from songsplat.cli.trainer_cli import run_train
        try:
            run_train(
                splatdata_paths=args.splatdata,
                output=args.output,
                architecture=args.arch,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch,
            )
        except Exception as e:
            _die(str(e))

    elif args.command == "reinforce":
        from songsplat.cli.reinforce import run_reinforce
        try:
            run_reinforce(
                splat_path=args.splat_file,
                chunk_paths=args.chunk_files,
            )
        except Exception as e:
            _die(str(e))

    elif args.command == "test":
        from songsplat.cli.runner import run_test
        try:
            run_test(splat_path=args.splat_file, input_path=args.input)
        except Exception as e:
            _die(str(e))

    elif args.command == "gui":
        from songsplat.ui.app import run
        run()


def _die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
