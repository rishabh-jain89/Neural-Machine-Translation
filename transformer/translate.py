import argparse
import time
from pathlib import Path

import torch

from evaluate_transformer import (
    load_model,
    greedy,
    beam,
    clean,
)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


class LiveTranslator:
    def __init__(self, project_root, checkpoint, device, beam_size=2, length_penalty=1.0, max_new_tokens=256, use_amp=True):
        (
            self.model,
            self.config,
            self.tokenizer_en,
            self.tokenizer_hi,
            self.checkpoint_path,
            _payload,
        ) = load_model(
            project_root,
            checkpoint,
            device,
        )

        self.device = device
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        self.max_new_tokens = max_new_tokens
        self.use_amp = bool(use_amp and device.type == "cuda")

        self.pad_src = self.tokenizer_en.pad_id()
        self.pad_tgt = self.tokenizer_hi.pad_id()
        self.bos_tgt = self.tokenizer_hi.bos_id()
        self.eos_tgt = self.tokenizer_hi.eos_id()
        self.eos_src = self.tokenizer_en.eos_id()

    def _prepare_source(self, text):
        text = text.strip()

        if not text:
            raise ValueError("Input sentence is empty.")

        content_ids = self.tokenizer_en.encode_as_ids(text)
        required_positions = len(content_ids) + 1

        max_position = getattr(self.config, "max_position", 512)

        if required_positions > max_position:
            raise ValueError(
                f"Input needs {required_positions} source positions, "
                f"but model max_position={max_position}. "
                "Please split it into shorter sentences."
            )

        src = torch.tensor(
            [content_ids + [self.eos_src]],
            dtype=torch.long,
            device=self.device,
        )

        return src, len(content_ids)

    def _decode_output(self, generated):
        row = generated[0].detach().cpu().tolist()

        ids = clean(
            row,
            self.bos_tgt,
            self.eos_tgt,
            self.pad_tgt,
        )

        text = self.tokenizer_hi.decode_ids(ids)

        return text, len(ids)

    @torch.inference_mode()
    def translate_greedy(self, text):
        src, source_tokens = self._prepare_source(text)

        synchronize(self.device)
        start = time.perf_counter()

        generated = greedy(
            self.model,
            src,
            self.pad_src,
            self.pad_tgt,
            self.bos_tgt,
            self.eos_tgt,
            self.max_new_tokens,
            self.use_amp,
        )

        synchronize(self.device)
        elapsed = time.perf_counter() - start

        translation, generated_tokens = self._decode_output(generated)

        return {
            "translation": translation,
            "source_tokens": source_tokens,
            "generated_tokens": generated_tokens,
            "milliseconds": elapsed * 1000.0,
        }

    @torch.inference_mode()
    def translate_beam(self, text):
        src, source_tokens = self._prepare_source(text)

        synchronize(self.device)
        start = time.perf_counter()

        generated = beam(
            self.model,
            src,
            self.pad_src,
            self.pad_tgt,
            self.bos_tgt,
            self.eos_tgt,
            self.beam_size,
            self.length_penalty,
            self.max_new_tokens,
            self.use_amp,
        )

        synchronize(self.device)
        elapsed = time.perf_counter() - start

        translation, generated_tokens = self._decode_output(generated)

        return {
            "translation": translation,
            "source_tokens": source_tokens,
            "generated_tokens": generated_tokens,
            "milliseconds": elapsed * 1000.0,
        }


def print_result(label, result):
    print()
    print(label)
    print("-" * len(label))
    print(result["translation"])

    print(
        f"[source tokens={result['source_tokens']} | "
        f"generated tokens={result['generated_tokens']} | "
        f"latency={result['milliseconds']:.1f} ms]"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Live English-to-Hindi Refined Transformer demo."
    )

    parser.add_argument(
        "--project-root",
        default=None,
        help="Project root. Default: parent of transformer/.",
    )

    parser.add_argument(
        "--checkpoint",
        default="averaged_5_dev_chrf.pt",
        help="Checkpoint filename or absolute path.",
    )

    parser.add_argument(
        "--method",
        choices=["beam", "greedy", "compare"],
        default="beam",
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--length-penalty",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--text",
        default=None,
        help="Translate one sentence and exit.",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent

    project_root = (
        Path(args.project_root).resolve()
        if args.project_root
        else script_dir.parent
    )

    if args.cpu:
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Run with --cpu if required."
            )
        device = torch.device("cuda:0")

    translator = LiveTranslator(
        project_root=project_root,
        checkpoint=args.checkpoint,
        device=device,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        max_new_tokens=args.max_new_tokens,
        use_amp=not args.no_amp,
    )

    print("=" * 72)
    print("LIVE ENGLISH -> HINDI TRANSLATOR")
    print("=" * 72)
    print(f"Device      : {device}")

    if device.type == "cuda":
        print(f"GPU         : {torch.cuda.get_device_name(0)}")

    print(f"Checkpoint  : {translator.checkpoint_path}")
    print(f"Parameters  : {sum(p.numel() for p in translator.model.parameters() if p.requires_grad):,}")
    print(
        f"Default     : {args.method} | "
        f"beam={args.beam_size} | "
        f"alpha={args.length_penalty}"
    )

    def run_translation(text, method):
        if method == "greedy":
            print_result(
                "Hindi — Greedy",
                translator.translate_greedy(text),
            )

        elif method == "beam":
            print_result(
                f"Hindi — Beam-{translator.beam_size}",
                translator.translate_beam(text),
            )

        else:
            print_result(
                "Hindi — Greedy",
                translator.translate_greedy(text),
            )
            print_result(
                f"Hindi — Beam-{translator.beam_size}",
                translator.translate_beam(text),
            )

    if args.text is not None:
        print()
        print("English")
        print("-------")
        print(args.text)
        run_translation(args.text, args.method)
        return

    current_method = args.method

    print(
        "\nType an English sentence and press Enter.\n"
        "Commands:\n"
        "  :beam     use Beam-2 decoding\n"
        "  :greedy   use greedy decoding\n"
        "  :compare  show both\n"
        "  :quit     exit\n"
    )

    while True:
        try:
            text = input("English > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not text:
            continue

        if text == ":quit":
            print("Exiting.")
            break

        if text == ":beam":
            current_method = "beam"
            print("Decoding -> beam")
            continue

        if text == ":greedy":
            current_method = "greedy"
            print("Decoding -> greedy")
            continue

        if text == ":compare":
            current_method = "compare"
            print("Decoding -> compare")
            continue

        try:
            run_translation(text, current_method)
        except Exception as exc:
            print(f"\nTranslation error: {exc}")

        print()


if __name__ == "__main__":
    main()
