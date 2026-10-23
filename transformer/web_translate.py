import argparse
import threading
import time
from pathlib import Path

import torch
from flask import Flask, jsonify, request, render_template_string

from evaluate_transformer import (
    load_model,
    greedy,
    beam,
    clean,
)


HTML = r"""
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <title>Neural Machine Translation</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            min-height: 100vh;
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            color: #222;
        }

        .container {
            width: min(900px, calc(100% - 40px));
            margin: 60px auto;
        }

        h1 {
            text-align: center;
            margin-bottom: 40px;
            font-size: 32px;
            font-weight: 600;
        }

        .translation-box {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        .language-box {
            display: flex;
            flex-direction: column;
        }

        label {
            margin-bottom: 8px;
            font-size: 16px;
            font-weight: 600;
        }

        textarea {
            width: 100%;
            height: 220px;
            padding: 15px;
            border: 1px solid #cccccc;
            border-radius: 6px;
            background: white;
            color: #222;
            font-size: 17px;
            line-height: 1.6;
            resize: vertical;
            outline: none;
        }

        textarea:focus {
            border-color: #666666;
        }

        #hindi {
            font-family:
                "Noto Sans Devanagari",
                "Nirmala UI",
                "Mangal",
                sans-serif;
        }

        .controls {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 12px;
            margin-top: 25px;
        }

        select {
            padding: 11px 14px;
            border: 1px solid #cccccc;
            border-radius: 5px;
            background: white;
            color: #222;
            font-size: 15px;
            cursor: pointer;
        }

        button {
            padding: 11px 24px;
            border: none;
            border-radius: 5px;
            background: #333333;
            color: white;
            font-size: 15px;
            cursor: pointer;
        }

        button:hover {
            background: #111111;
        }

        button:disabled {
            background: #888888;
            cursor: not-allowed;
        }

        @media (max-width: 700px) {
            .translation-box {
                grid-template-columns: 1fr;
            }

            .container {
                margin-top: 30px;
            }

            textarea {
                height: 180px;
            }

            .controls {
                flex-direction: column;
            }

            select,
            button {
                width: 100%;
            }
        }
    </style>
</head>

<body>

    <div class="container">

        <h1>Neural Machine Translation</h1>

        <div class="translation-box">

            <div class="language-box">
                <label for="english">English</label>

                <textarea
                    id="english"
                    placeholder="Enter English text..."
                    autofocus
                ></textarea>
            </div>

            <div class="language-box">
                <label for="hindi">Hindi</label>

                <textarea
                    id="hindi"
                    placeholder="Hindi translation..."
                    readonly
                ></textarea>
            </div>

        </div>

        <div class="controls">

            <select id="method">
                <option value="beam" selected>
                    Refined Transformer — Beam
                </option>

                <option value="greedy">
                    Refined Transformer — Greedy
                </option>
            </select>

            <button
                id="translateBtn"
                onclick="translateText()"
            >
                Translate
            </button>

        </div>

    </div>


    <script>

        async function translateText() {

            const source =
                document.getElementById("english")
                .value
                .trim();

            const method =
                document.getElementById("method").value;

            const output =
                document.getElementById("hindi");

            const button =
                document.getElementById("translateBtn");


            if (!source) {
                alert("Please enter English text.");
                return;
            }


            button.disabled = true;
            button.textContent = "Translating...";


            try {

                const response = await fetch(
                    "/api/translate",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type": "application/json"
                        },

                        body: JSON.stringify({
                            text: source,
                            method: method
                        })
                    }
                );


                const data = await response.json();


                if (!response.ok) {
                    throw new Error(
                        data.error || "Translation failed."
                    );
                }


                output.value = data.translation;

            }

            catch (error) {

                output.value = "";

                alert(error.message);

            }

            finally {

                button.disabled = false;

                button.textContent = "Translate";

            }
        }


        document
            .getElementById("english")
            .addEventListener(
                "keydown",
                function(event) {

                    if (
                        (event.ctrlKey || event.metaKey)
                        && event.key === "Enter"
                    ) {
                        translateText();
                    }
                }
            );

    </script>

</body>
</html>
"""


class Translator:
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

        self.lock = threading.Lock()

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _prepare_source(self, text):
        text = text.strip()

        if not text:
            raise ValueError("Input sentence is empty.")

        content_ids = self.tokenizer_en.encode_as_ids(text)
        needed = len(content_ids) + 1

        max_position = getattr(
            self.config,
            "max_position",
            512,
        )

        if needed > max_position:
            raise ValueError(
                f"Input is too long ({needed} source positions). "
                f"Model limit is {max_position}. "
                "Please split it into shorter sentences."
            )

        src = torch.tensor(
            [content_ids + [self.eos_src]],
            dtype=torch.long,
            device=self.device,
        )

        return src, len(content_ids)

    def _decode(self, generated):
        ids = generated[0].detach().cpu().tolist()

        ids = clean(
            ids,
            self.bos_tgt,
            self.eos_tgt,
            self.pad_tgt,
        )

        return self.tokenizer_hi.decode_ids(ids), len(ids)

    @torch.inference_mode()
    def translate(self, text, method="beam"):
        src, source_tokens = self._prepare_source(text)

        with self.lock:
            self._sync()
            start = time.perf_counter()

            if method == "greedy":
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

                method_name = "Greedy"

            elif method == "beam":
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

                method_name = (
                    f"Beam-{self.beam_size}, "
                    f"α={self.length_penalty}"
                )

            else:
                raise ValueError(
                    "method must be 'beam' or 'greedy'"
                )

            self._sync()
            elapsed = time.perf_counter() - start

        translation, generated_tokens = self._decode(generated)

        return {
            "translation": translation,
            "method": method_name,
            "source_tokens": source_tokens,
            "generated_tokens": generated_tokens,
            "latency_ms": elapsed * 1000.0,
        }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-root",
        default=None,
    )

    parser.add_argument(
        "--checkpoint",
        default="averaged_5_dev_chrf.pt",
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Bind address. Keep 127.0.0.1 when using SSH "
            "port forwarding. Use 0.0.0.0 only on a trusted network."
        ),
    )

    parser.add_argument(
        "--port",
        type=int,
        default=7860,
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
                "CUDA is unavailable. Use --cpu if necessary."
            )
        device = torch.device("cuda:0")

    print("Loading translation model...", flush=True)

    translator = Translator(
        project_root=project_root,
        checkpoint=args.checkpoint,
        device=device,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        max_new_tokens=args.max_new_tokens,
        use_amp=not args.no_amp,
    )

    params = sum(
        p.numel()
        for p in translator.model.parameters()
        if p.requires_grad
    )

    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = "CPU"

    print("=" * 72, flush=True)
    print("ENGLISH -> HINDI WEB TRANSLATOR", flush=True)
    print("=" * 72, flush=True)
    print(f"Device     : {device_name}", flush=True)
    print(f"Checkpoint : {translator.checkpoint_path}", flush=True)
    print(f"Parameters : {params:,}", flush=True)
    print(
        f"Decoder    : Beam-{args.beam_size}, "
        f"alpha={args.length_penalty}",
        flush=True,
    )
    print(
        f"Serving    : http://{args.host}:{args.port}",
        flush=True,
    )
    print("=" * 72, flush=True)

    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template_string(
            HTML,
            model_name=Path(
                translator.checkpoint_path
            ).name,
            params=f"{params:,}",
            device_name=device_name,
        )

    @app.post("/api/translate")
    def api_translate():
        try:
            payload = request.get_json(
                silent=True
            ) or {}

            text = str(
                payload.get("text", "")
            ).strip()

            method = str(
                payload.get("method", "beam")
            ).strip().lower()

            if not text:
                return jsonify(
                    error="Please enter an English sentence."
                ), 400

            if method not in {"beam", "greedy"}:
                return jsonify(
                    error="Invalid decoding method."
                ), 400

            result = translator.translate(
                text,
                method=method,
            )

            return jsonify(result)

        except Exception as exc:
            print(
                f"Translation error: {exc}",
                flush=True,
            )

            return jsonify(
                error=str(exc)
            ), 500

    app.run(
        host=args.host,
        port=args.port,
        debug=False,
        threaded=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
