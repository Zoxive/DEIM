"""
Catch the batch that turns the detector's predictions non-finite, and say where in the network it happened.

Why this exists. A training config died on two runs of three with NaN *predictions*
reaching the Hungarian matcher (`generalized_box_iou`'s corner assertion), after a clean loss curve. The
`NaN.pth` the loop already dumps showed every learned weight finite and only BatchNorm running stats
poisoned from `encoder.lateral_convs.0` onward, with `encoder.input_proj` clean. So a forward pass on
finite weights overflowed somewhere between the input projection and the first lateral conv - which is
the AIFI transformer layer, the one place in that span that carries no BN stats to leave a trace in.
Nothing recorded which photographs were in the batch, and `--seed` has never seeded anything here (the
`setup_distributed` call is commented out in `train.py`), so the shuffle cannot be replayed to find them.

This does the recording instead, at the moment it happens:

- every step: the peak |activation| going into and out of the AIFI encoder and the peak attention logit
  are appended to `<output_dir>/nonfinite/steps.jsonl`, with the batch's image ids when the peak is within
  `WARN_FRACTION` of the fp16 ceiling - near misses rank the dangerous photographs even on a run that
  never dies;
- on a non-finite prediction: the batch is written (`e<E>_s<S>.pt` + a PNG per sample), the BN running
  stats are rolled back to their pre-forward values, the forward is re-run with a hook on every module
  to name the first one whose output is non-finite, re-run again in fp32 to say whether it is fp16
  overflow at all, and each sample is run alone to say which one does it. Then the step is skipped.

Skipping is loud and capped: `max_skips` events and the run raises. A guard that quietly trains through
every bad batch would be a different recipe; this one exists to find the batch, not to hide it.
"""

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

FP16_MAX = 65504.0
# Log the batch's ids whenever any watched peak reaches this fraction of the fp16 ceiling.
WARN_FRACTION = 0.25


def _amax(x):
    if torch.is_tensor(x) and x.is_floating_point():
        return x.detach().abs().amax().float() if x.numel() else None
    if isinstance(x, (list, tuple)):
        vals = [_amax(v) for v in x]
        vals = [v for v in vals if v is not None]
        return torch.stack(vals).amax() if vals else None
    if isinstance(x, dict):
        return _amax(list(x.values()))
    return None


def _finite(x):
    if torch.is_tensor(x) and x.is_floating_point():
        return bool(torch.isfinite(x).all())
    if isinstance(x, (list, tuple)):
        return all(_finite(v) for v in x)
    if isinstance(x, dict):
        return all(_finite(v) for v in x.values())
    return True


class NonFiniteGuard:
    def __init__(self, model: nn.Module, output_dir, max_skips: int = 10):
        self.model = model
        self.dir = Path(output_dir) / 'nonfinite'
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_skips = max_skips
        self.skips = 0
        self.steps_file = open(self.dir / 'steps.jsonl', 'a', buffering=1)

        self.buffers = [b for _, b in model.named_buffers() if b is not None]
        self.snapshot_bufs = [b.detach().clone() for b in self.buffers]

        enc = model.encoder
        self.aifi = enc.encoder[0]
        self.attn = self.aifi.layers[0].self_attn
        self._peaks = {}
        self._hooks = [
            self.aifi.register_forward_pre_hook(self._pre('aifi_in')),
            self.aifi.register_forward_hook(self._post('aifi_out')),
            self.attn.register_forward_pre_hook(self._attn_logits, with_kwargs=True),
        ]
        print(f'[nonfinite-guard] armed: {len(self.buffers)} buffers snapshotted per step, '
              f'max {max_skips} skips, log {self.dir}')

    # --- per-step telemetry -------------------------------------------------------------------------

    def _pre(self, key):
        def hook(_m, args):
            self._peaks[key] = _amax(args[0])
        return hook

    def _post(self, key):
        def hook(_m, _args, out):
            self._peaks[key] = _amax(out)
        return hook

    def _attn_logits(self, m: nn.MultiheadAttention, args, kwargs):
        # Recompute the pre-softmax logits in fp32 from the same inputs: this is the number an fp16
        # matmul has to hold, and nothing else in the module exposes it.
        q_in = args[0] if len(args) > 0 else kwargs['query']
        k_in = args[1] if len(args) > 1 else kwargs['key']
        with torch.no_grad(), torch.autocast(device_type=q_in.device.type, enabled=False):
            e = m.embed_dim
            w, b = m.in_proj_weight.float(), m.in_proj_bias.float()
            q = q_in.float() @ w[:e].T + b[:e]
            k = k_in.float() @ w[e:2 * e].T + b[e:2 * e]
            B, L, _ = q.shape
            h, d = m.num_heads, e // m.num_heads
            q = q.view(B, L, h, d).transpose(1, 2) / math.sqrt(d)
            k = k.view(B, L, h, d).transpose(1, 2)
            self._peaks['attn_logit'] = (q @ k.transpose(-1, -2)).abs().amax()
            self._peaks['attn_q'] = q.abs().amax() * math.sqrt(d)

    def snapshot(self):
        torch._foreach_copy_(self.snapshot_bufs, self.buffers)

    def restore(self):
        torch._foreach_copy_(self.buffers, self.snapshot_bufs)

    def log_step(self, epoch, step, targets, pred_boxes_finite, loss_value=None):
        peaks = {k: (float(v) if v is not None else None) for k, v in self._peaks.items()}
        row = {'epoch': epoch, 'step': step, 'finite': pred_boxes_finite, **peaks}
        if loss_value is not None:
            row['loss'] = loss_value
        if (not pred_boxes_finite) or any(v is not None and (not math.isfinite(v) or v >= WARN_FRACTION * FP16_MAX)
                                          for v in peaks.values()):
            row['image_ids'] = [int(t['image_id'].reshape(-1)[0]) for t in targets]
        self.steps_file.write(json.dumps(row) + '\n')
        self._peaks.clear()

    # --- the event ----------------------------------------------------------------------------------

    def handle(self, samples, targets, epoch, step, amp_dtype, dataset=None):
        """Called with BN buffers already polluted by the bad forward. Records, diagnoses, restores."""
        t0 = time.time()
        self.skips += 1
        tag = f'e{epoch:02d}_s{step:05d}'
        print(f'\n!!!!! [nonfinite-guard] non-finite predictions at epoch {epoch} step {step} '
              f'(event {self.skips}/{self.max_skips}) - diagnosing, see {self.dir}/{tag}.json', flush=True)

        ids = [int(t['image_id'].reshape(-1)[0]) for t in targets]
        files = self._file_names(dataset, ids)
        torch.save({'samples': samples.detach().cpu(), 'targets': [{k: v.cpu() for k, v in t.items()} for t in targets],
                    'image_ids': ids, 'files': files, 'epoch': epoch, 'step': step}, self.dir / f'{tag}.pt')
        self._write_pngs(samples, targets, tag)

        report = {'epoch': epoch, 'step': step, 'image_ids': ids, 'files': files,
                  'input_amax': float(samples.abs().amax()), 'input_finite': bool(torch.isfinite(samples).all())}

        was_training = self.model.training
        try:
            self.restore()
            report['trace_amp'] = self._trace(samples, targets, amp_dtype)
            self.restore()
            report['trace_fp32'] = self._trace(samples, targets, None, keep=6)
            self.restore()
            report['per_sample_eval_amp'] = self.attach_ids(self._per_sample(samples, amp_dtype), ids, files)
            report['per_sample_eval_fp32'] = self.attach_ids(self._per_sample(samples, None), ids, files)
        finally:
            self.model.train(was_training)
            self.restore()
            self._peaks.clear()

        report['seconds'] = round(time.time() - t0, 1)
        (self.dir / f'{tag}.json').write_text(json.dumps(report, indent=2))
        first = report['trace_amp'].get('first_nonfinite')
        culprits = [r for r in report['per_sample_eval_amp'] if not r['finite']]
        print(f'!!!!! [nonfinite-guard] first non-finite module under autocast: {first}', flush=True)
        print(f'!!!!! [nonfinite-guard] fp32 re-run finite: {report["trace_fp32"]["output_finite"]}', flush=True)
        print(f'!!!!! [nonfinite-guard] samples non-finite alone (eval, autocast): '
              f'{[(c["index"], c["image_id"], c["file"]) for c in culprits]}', flush=True)

        if self.skips >= self.max_skips:
            raise RuntimeError(f'[nonfinite-guard] {self.skips} non-finite batches - over the cap of '
                               f'{self.max_skips}; this is not a stray batch. Reports in {self.dir}')

    def _file_names(self, dataset, ids):
        coco = getattr(dataset, 'coco', None)
        if coco is None:
            return [None] * len(ids)
        return [coco.imgs.get(i, {}).get('file_name') for i in ids]

    def _write_pngs(self, samples, targets, tag):
        try:
            from PIL import Image
        except ImportError:
            return
        out = self.dir / tag
        out.mkdir(exist_ok=True)
        for i, img in enumerate(samples.detach().float().cpu()):
            arr = torch.nan_to_num(img, nan=1.0, posinf=1.0, neginf=0.0).clamp(0, 1).mul(255).byte()
            Image.fromarray(arr.permute(1, 2, 0).numpy()).save(out / f'{i:02d}_id{int(targets[i]["image_id"].reshape(-1)[0])}.png')

    @torch.no_grad()
    def _trace(self, samples, targets, amp_dtype, keep=40):
        """Forward with a hook on every module; name the first one (in call order) whose output is non-finite."""
        order = []
        handles = []

        def make(name):
            def hook(_m, args, out):
                a_in, a_out = _amax(list(args)), _amax(out)
                order.append({'module': name,
                              'in_amax': None if a_in is None else float(a_in),
                              'out_amax': None if a_out is None else float(a_out),
                              'in_finite': _finite(list(args)), 'out_finite': _finite(out)})
            return hook

        for name, m in self.model.named_modules():
            if name:
                handles.append(m.register_forward_hook(make(name)))
        try:
            self.model.train()
            if amp_dtype is not None:
                with torch.autocast(device_type=samples.device.type, dtype=amp_dtype):
                    out = self.model(samples, targets=targets)
            else:
                with torch.autocast(device_type=samples.device.type, enabled=False):
                    out = self.model(samples.float(), targets=targets)
        finally:
            for h in handles:
                h.remove()

        first = next((i for i, r in enumerate(order) if not r['out_finite']), None)
        result = {'dtype': str(amp_dtype), 'output_finite': _finite(out.get('pred_boxes')),
                  'first_nonfinite': None if first is None else order[first]['module'],
                  'peaks': dict(sorted(((k, float(v)) for k, v in self._peaks.items() if v is not None)))}
        if first is not None:
            result['window'] = order[max(0, first - keep // 2): first + 3]
        else:
            result['largest'] = sorted((r for r in order if r['out_amax'] is not None),
                                       key=lambda r: -r['out_amax'])[:keep]
        return result

    @torch.no_grad()
    def _per_sample(self, samples, amp_dtype):
        self.model.eval()
        rows = []
        for i in range(samples.shape[0]):
            x = samples[i:i + 1]
            if amp_dtype is not None:
                with torch.autocast(device_type=x.device.type, dtype=amp_dtype):
                    out = self.model(x)
            else:
                with torch.autocast(device_type=x.device.type, enabled=False):
                    out = self.model(x.float())
            peaks = {k: float(v) for k, v in self._peaks.items() if v is not None}
            self._peaks.clear()
            rows.append({'index': i, 'finite': _finite(out.get('pred_boxes')), **peaks})
        return rows

    def attach_ids(self, rows, ids, files):
        for r in rows:
            r['image_id'] = ids[r['index']]
            r['file'] = files[r['index']]
        return rows
