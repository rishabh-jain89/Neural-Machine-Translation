import argparse, csv, json, math, time
from pathlib import Path
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sacrebleu.metrics import BLEU, CHRF, TER
from train_transformer import TransformerConfig, build_model

def read_lines(p):
    with open(p, 'r', encoding='utf-8') as f:
        return [x.rstrip('\n') for x in f]

def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()

def metrics(hyp, ref):
    b = BLEU()
    c = CHRF(word_order=2)
    t = TER()
    return {'BLEU': b.corpus_score(hyp, [ref]).score, 'chrF++': c.corpus_score(hyp, [ref]).score, 'TER': t.corpus_score(hyp, [ref]).score, 'BLEU_signature': str(b.get_signature()), 'chrF_signature': str(c.get_signature()), 'TER_signature': str(t.get_signature())}

class EvalDataset(Dataset):

    def __init__(self, en, hi, te, th):
        assert len(en) == len(hi)
        self.en_texts = en
        self.hi_texts = hi
        self.x = [te.encode_as_ids(s) for s in en]
        self.y = [th.encode_as_ids(s) for s in hi]
        self.eos_src = te.eos_id()
        self.bos = th.bos_id()
        self.eos = th.eos_id()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        src = torch.tensor(self.x[i] + [self.eos_src], dtype=torch.long)
        tin = torch.tensor([self.bos] + self.y[i], dtype=torch.long)
        tout = torch.tensor(self.y[i] + [self.eos], dtype=torch.long)
        return (i, src, tin, tout)

class Collate:

    def __init__(self, ps, pt):
        self.ps = ps
        self.pt = pt

    def __call__(self, b):
        idx = torch.tensor([x[0] for x in b])
        src = pad_sequence([x[1] for x in b], batch_first=True, padding_value=self.ps)
        tin = pad_sequence([x[2] for x in b], batch_first=True, padding_value=self.pt)
        tout = pad_sequence([x[3] for x in b], batch_first=True, padding_value=self.pt)
        return (idx, src, tin, tout)

def loader(ds, collate, bs):
    return DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate, num_workers=0, pin_memory=torch.cuda.is_available())

def load_model(root, checkpoint, device, run_name="refined_transformer"):
    checkpoint_path = Path(checkpoint)

    if not checkpoint_path.is_absolute():
        direct_path = root / checkpoint_path

        run_path = (root / "output" / "checkpoints" / f"{run_name}_seed19" / checkpoint_path)
        legacy_path = (root / "output" / "checkpoints" / "refined_transformer_seed19" / checkpoint_path)

        if direct_path.exists():
            checkpoint_path = direct_path
        elif run_path.exists():
            checkpoint_path = run_path
        elif legacy_path.exists():
            checkpoint_path = legacy_path
        else:
            raise FileNotFoundError(
                "Checkpoint not found. Checked:\n"
                f"  {direct_path}\n"
                f"  {run_path}\n"
                f"  {legacy_path}"
            )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config_values = payload["config"]

    config_values = dict(config_values)

    if "final_norm" not in config_values:
        config_values["final_norm"] = bool(config_values.get("norm_first", False))

    if "name" not in config_values:
        config_values["name"] = run_name

    valid_fields = TransformerConfig.__dataclass_fields__

    config = TransformerConfig(
        **{
            key: value
            for key, value in config_values.items()
            if key in valid_fields
        }
    )

    tokenizer_en = spm.SentencePieceProcessor(model_file=str(root / "output/models/en_unigram_500000_16000_0.9999.model"))

    tokenizer_hi = spm.SentencePieceProcessor(
        model_file=str(root / "output/models/hi_unigram_500000_16000_0.9999.model"))

    data = {
        "vocab_size_en": tokenizer_en.vocab_size(),
        "vocab_size_hi": tokenizer_hi.vocab_size(),
        "pad_id_en": tokenizer_en.pad_id(),
        "pad_id_hi": tokenizer_hi.pad_id(),
    }

    model = build_model(data, config, device)

    state_dict = payload["model_state_dict"]

    if ("src_pos.pe" in state_dict and "src_position.pe" not in state_dict):
        state_dict["src_position.pe"] = (state_dict.pop("src_pos.pe"))

    if ("tgt_pos.pe" in state_dict and "tgt_position.pe" not in state_dict):
        state_dict["tgt_position.pe"] = (state_dict.pop("tgt_pos.pe"))

    model.load_state_dict(state_dict, strict=True)

    model.eval()

    return (model, config, tokenizer_en, tokenizer_hi, checkpoint_path, payload)

@torch.inference_mode()


def teacher_forced(model, ldr, device, ps, pt, use_amp):
    loss_fn = nn.CrossEntropyLoss(ignore_index=pt)
    tl = torch.zeros((), device=device, dtype=torch.float64)
    nt = torch.zeros((), device=device, dtype=torch.long)
    cor = torch.zeros((), device=device, dtype=torch.long)

    for _, src, tin, tout in ldr:
        src = src.to(device)
        tin = tin.to(device)
        tout = tout.to(device)
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(src, tin, src.eq(ps), tin.eq(pt))
            flat = logits.reshape(-1, logits.size(-1))
            targets = tout.reshape(-1)
            loss = loss_fn(flat, targets)
        mask = targets.ne(pt)
        n = mask.sum()
        tl += loss.double() * n
        nt += n
        cor += ((flat.argmax(-1) == targets) & mask).sum()

    ce = (tl / nt).item()
    return {'test_cross_entropy': ce, 'test_perplexity': math.exp(ce), 'test_token_accuracy': (cor.double() / nt * 100).item()}

@torch.inference_mode()

def greedy(model, src, ps, pt, bos, eos, max_new, use_amp):
    sp = src.eq(ps)

    with torch.amp.autocast('cuda', enabled=use_amp):
        mem = model.encode(src, sp)

    seq = torch.full((src.size(0), 1), bos, dtype=torch.long, device=src.device)
    done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)

    for _ in range(max_new):
        with torch.amp.autocast('cuda', enabled=use_amp):
            dec = model.decode(seq, mem, seq.eq(pt), sp)
            nxt = model.output_projection(dec[:, -1]).argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, pt), nxt)
        seq = torch.cat([seq, nxt[:, None]], 1)
        done |= nxt.eq(eos)
        if bool(done.all()):
            break

    return seq

@torch.inference_mode()

def beam(model, src, ps, pt, bos, eos, beam_size, alpha, max_new, use_amp):

    B = src.size(0)
    sp = src.eq(ps)

    with torch.amp.autocast('cuda', enabled=use_amp):
        mem = model.encode(src, sp)

    S, D = (mem.size(1), mem.size(2))
    mem = mem[:, None].expand(B, beam_size, S, D).reshape(B * beam_size, S, D)
    esp = sp[:, None].expand(B, beam_size, sp.size(1)).reshape(B * beam_size, sp.size(1))
    seq = torch.full((B, beam_size, 1), bos, dtype=torch.long, device=src.device)
    scores = torch.full((B, beam_size), float('-inf'), device=src.device)

    scores[:, 0] = 0
    done = torch.zeros((B, beam_size), dtype=torch.bool, device=src.device)
    lengths = torch.zeros((B, beam_size), dtype=torch.long, device=src.device)
    V = model.output_projection.out_features

    for _ in range(max_new):
        flat = seq.reshape(B * beam_size, -1)
        with torch.amp.autocast('cuda', enabled=use_amp):
            dec = model.decode(flat, mem, flat.eq(pt), esp)
            lp = torch.log_softmax(model.output_projection(dec[:, -1]).float(), -1).reshape(B, beam_size, V)

        if bool(done.any()):
            lp = torch.where(done[:, :, None], torch.full_like(lp, float('-inf')), lp)
            eos_col = lp[:, :, eos]
            eos_col = torch.where(done, torch.zeros_like(eos_col), eos_col)
            lp[:, :, eos] = eos_col

        cand = (scores[:, :, None] + lp).reshape(B, -1)
        ns, ni = torch.topk(cand, beam_size, -1)
        parent = ni // V
        tok = ni % V

        parent_seq = torch.gather(seq, 1, parent[:, :, None].expand(-1, -1, seq.size(-1)))
        pdone = torch.gather(done, 1, parent)
        plen = torch.gather(lengths, 1, parent)
        seq = torch.cat([parent_seq, tok[:, :, None]], -1)
        lengths = plen + (~pdone).long()
        done = pdone | tok.eq(eos)
        scores = ns

        if bool(done.all()):
            break

    norm = ((5.0 + lengths.clamp_min(1).float()) / 6.0) ** alpha
    best = (scores / norm).argmax(1)

    return seq[torch.arange(B, device=src.device), best]

def clean(row, bos, eos, pad):
    out = []
    for t in row:
        t = int(t)
        if t == bos:
            continue
        if t == eos:
            break
        if t != pad:
            out.append(t)
    return out

def generate(model, ds, collate, th, device, method, bs, ps, pt, bos, eos, max_new, use_amp, beam_size=4, alpha=0.8):
    ldr = loader(ds, collate, bs)
    hyp = [None] * len(ds)
    sync(device)
    start = time.perf_counter()

    for bi, (idx, src, _, _) in enumerate(ldr, 1):
        src = src.to(device)
        out = greedy(model, src, ps, pt, bos, eos, max_new, use_amp) if method == 'greedy' else beam(model, src, ps, pt, bos, eos, beam_size, alpha, max_new, use_amp)
        for row, i in zip(out.cpu().tolist(), idx.tolist()):
            hyp[i] = th.decode_ids(clean(row, bos, eos, pt))

        if bi % 20 == 0 or bi == len(ldr):
            print(f'{method}: {min(bi * bs, len(ds)):,}/{len(ds):,}')

    sync(device)
    return (hyp, time.perf_counter() - start)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['tune-beam', 'test'], required=True)
    p.add_argument('--checkpoint', default='averaged_5_dev_chrf.pt')
    p.add_argument("--run-name", default="refined_transformer")
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--beam-sizes', type=int, nargs='+', default=[2, 4, 6])
    p.add_argument('--length-penalties', type=float, nargs='+', default=[0.6, 0.8, 1.0])
    p.add_argument('--beam-size', type=int, default=None)
    p.add_argument('--length-penalty', type=float, default=None)
    return p.parse_args()

def main():
    a = parse_args()
    root = Path(__file__).resolve().parent.parent

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    
    device = torch.device('cuda:0')
    use_amp = True
    model, cfg, te, th, cp, payload = load_model(root, a.checkpoint, device, run_name=a.run_name)

    ps = te.pad_id()
    pt = th.pad_id()
    bos = th.bos_id()
    eos = th.eos_id()

    outdir = root / f"output/evaluation/{a.run_name}"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Checkpoint: {cp}')
    print(f'Params: {sum((p.numel() for p in model.parameters() if p.requires_grad)):,}')

    if a.mode == 'tune-beam':
        en = read_lines(root / 'data/clean/dev.en')
        hi = read_lines(root / 'data/clean/dev.hi')
    else:
        en = read_lines(root / 'data/clean/test.en')
        hi = read_lines(root / 'data/clean/test.hi')

    ds = EvalDataset(en, hi, te, th)
    coll = Collate(ps, pt)

    if a.mode == 'tune-beam':
        results = []
        gh, gs = generate(model, ds, coll, th, device, 'greedy', a.batch_size, ps, pt, bos, eos, a.max_new_tokens, use_amp)
        gm = metrics(gh, hi)
        results.append({'method': 'greedy', 'beam_size': 1, 'length_penalty': None, 'seconds': gs, **gm})
        print(f"DEV GREEDY BLEU={gm['BLEU']:.4f} chrF++={gm['chrF++']:.4f} TER={gm['TER']:.4f}")

        for b in a.beam_sizes:
            for alpha in a.length_penalties:
                h, s = generate(model, ds, coll, th, device, 'beam', a.batch_size, ps, pt, bos, eos, a.max_new_tokens, use_amp, b, alpha)
                m = metrics(h, hi)
                r = {'method': 'beam', 'beam_size': b, 'length_penalty': alpha, 'seconds': s, **m}
                results.append(r)
                print(f"beam={b} alpha={alpha}: BLEU={m['BLEU']:.4f} chrF++={m['chrF++']:.4f} TER={m['TER']:.4f}")

        best = max([x for x in results if x['method'] == 'beam'], key=lambda x: x['chrF++'])
        out = {'checkpoint': str(cp), 'selection_metric': 'dev chrF++', 'best': best, 'all_results': results}
        path = outdir / 'beam_tuning_dev.json'
        json.dump(out, open(path, 'w'), indent=2)

        print(f"BEST: beam={best['beam_size']}, alpha={best['length_penalty']}, chrF++={best['chrF++']:.4f}")
        print(f'Saved: {path}')
    else:
        tune = outdir / 'beam_tuning_dev.json'

        if a.beam_size is not None and a.length_penalty is not None:
            b = a.beam_size
            alpha = a.length_penalty
        elif tune.exists():
            t = json.load(open(tune))
            b = int(t['best']['beam_size'])
            alpha = float(t['best']['length_penalty'])
        else:
            raise RuntimeError('Run --mode tune-beam first or pass --beam-size and --length-penalty')
        
        tf = teacher_forced(model, loader(ds, coll, 32), device, ps, pt, use_amp)
        gh, gs = generate(model, ds, coll, th, device, 'greedy', a.batch_size, ps, pt, bos, eos, a.max_new_tokens, use_amp)
        gm = metrics(gh, hi)
        bh, bs = generate(model, ds, coll, th, device, 'beam', a.batch_size, ps, pt, bos, eos, a.max_new_tokens, use_amp, b, alpha)
        bm = metrics(bh, hi)

        summary = {'checkpoint': str(cp), 'teacher_forced': tf, 'greedy': {**gm, 'seconds': gs}, 'beam': {'beam_size': b, 'length_penalty': alpha, **bm, 'seconds': bs}}
        spath = outdir / 'test_metrics.json'

        json.dump(summary, open(spath, 'w'), indent=2)
        ppath = outdir / 'test_predictions.csv'

        with open(ppath, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f)
            w.writerow(['id', 'source_en', 'reference_hi', 'greedy_hi', 'beam_hi'])
            for i, (s, r, g, h) in enumerate(zip(en, hi, gh, bh)):
                w.writerow([i, s, r, g, h])

        print(f"Test CE={tf['test_cross_entropy']:.4f} PPL={tf['test_perplexity']:.4f} TokAcc={tf['test_token_accuracy']:.2f}%")
        print(f"GREEDY BLEU={gm['BLEU']:.4f} chrF++={gm['chrF++']:.4f} TER={gm['TER']:.4f}")
        print(f"BEAM-{b} alpha={alpha}: BLEU={bm['BLEU']:.4f} chrF++={bm['chrF++']:.4f} TER={bm['TER']:.4f}")
        print(f'Summary: {spath}')
        print(f'Predictions: {ppath}')

if __name__ == '__main__':
    main()
