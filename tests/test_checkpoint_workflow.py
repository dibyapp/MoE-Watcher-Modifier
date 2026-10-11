"""Tiny CPU fixtures: no model downloads, prompts, or private checkpoint paths."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file, load_file
from moe_pruner.checkpoint import Checkpoint
from moe_pruner.planner import build_uniform_plan
from moe_pruner.pruner import prune_checkpoint
from moe_pruner.schema import SCHEMAS

ROOT = Path(__file__).resolve().parents[1]

class CheckpointWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / 'source'
        self.source.mkdir()
        config = {'model_type': 'mixtral', 'num_hidden_layers': 2,
                  'num_local_experts': 4, 'num_experts_per_tok': 2, 'hidden_size': 3}
        (self.source / 'config.json').write_text(json.dumps(config))
        self.tensors = {'model.embed_tokens.weight': torch.arange(15.).reshape(5, 3)}
        for layer in range(2):
            prefix = f'model.layers.{layer}.block_sparse_moe'
            self.tensors[prefix + '.gate.weight'] = torch.arange(12.).reshape(4, 3)
            for expert in range(4):
                for weight in ['w1', 'w2', 'w3']:
                    self.tensors[f'{prefix}.experts.{expert}.{weight}.weight'] = torch.full((3, 3), float(expert))
        save_file(self.tensors, str(self.source / 'model.safetensors'))
        self.checkpoint = Checkpoint(self.source)
        self.schema = SCHEMAS['mixtral']
        self.plan = build_uniform_plan(2, 4, 2, 2, new_topk=1)
        self.plan.layer_keep_ids = [[1, 3], [0, 2]]

    def snapshot(self):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.iterdir()}

    def test_rewrite_compacts_ids_slices_routers_and_preserves_source(self):
        before = self.snapshot()
        output = self.base / 'pruned'
        manifest = prune_checkpoint(self.checkpoint, self.schema, self.plan, output, verbose=False)
        got = load_file(str(output / 'model.safetensors'))
        for layer, keep in enumerate(self.plan.layer_keep_ids):
            prefix = f'model.layers.{layer}.block_sparse_moe'
            self.assertTrue(torch.equal(got[prefix + '.gate.weight'], self.tensors[prefix + '.gate.weight'][keep]))
            for new, old in enumerate(keep):
                self.assertTrue(torch.equal(got[f'{prefix}.experts.{new}.w1.weight'], self.tensors[f'{prefix}.experts.{old}.w1.weight']))
            self.assertNotIn(prefix + '.experts.2.w1.weight', got)
        self.assertTrue(torch.equal(got['model.embed_tokens.weight'], self.tensors['model.embed_tokens.weight']))
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(json.loads((output / 'config.json').read_text())['num_local_experts'], 2)
        self.assertEqual(json.loads((output / 'config.json').read_text())['num_experts_per_tok'], 1)
        self.assertEqual(len(Checkpoint(output).all_keys), len(got))
        self.assertEqual(manifest['total_output_keys'], len(got))

    def test_dry_run_writes_nothing(self):
        output = self.base / 'dry-run'
        before = self.snapshot()
        prune_checkpoint(self.checkpoint, self.schema, self.plan, output, dry_run=True, verbose=False)
        self.assertFalse(output.exists())
        self.assertEqual(self.snapshot(), before)

    def test_overlapping_and_nonempty_outputs_rejected(self):
        for output in [self.source, self.source / 'nested', self.base]:
            with self.assertRaises(ValueError):
                prune_checkpoint(self.checkpoint, self.schema, self.plan, output, verbose=False)
        occupied = self.base / 'occupied'; occupied.mkdir()
        (occupied / 'important.txt').write_text('keep')
        with self.assertRaises(ValueError):
            prune_checkpoint(self.checkpoint, self.schema, self.plan, occupied, verbose=False)
        self.assertEqual((occupied / 'important.txt').read_text(), 'keep')

    def test_invalid_plans_rejected(self):
        for ids in [[1, 1], [-1, 2], [0, 4], [0.0, 1]]:
            self.plan.layer_keep_ids[0] = ids
            with self.assertRaises(ValueError): self.plan.validate()
        self.plan.layer_keep_ids[0] = [1, 3]
        self.plan.target_topk = 3
        with self.assertRaises(ValueError): self.plan.validate()

    def test_wrong_config_and_unmatched_expert_layout_rejected_before_writes(self):
        output = self.base / 'rejected'
        self.checkpoint.config['num_local_experts'] = 8
        with self.assertRaises(ValueError):
            prune_checkpoint(self.checkpoint, self.schema, self.plan, output, verbose=False)
        self.assertFalse(output.exists())
        self.checkpoint.config['num_local_experts'] = 4
        self.checkpoint.weight_map = {k: v for k, v in self.checkpoint.weight_map.items() if '.experts.' not in k}
        with self.assertRaises(ValueError):
            prune_checkpoint(self.checkpoint, self.schema, self.plan, output, verbose=False)
        self.assertFalse(output.exists())

    def test_documented_cli_pipeline(self):
        report = self.base / 'stats' / 'report.json'
        plan = report.with_name('report-plan.json')
        commands = [
            ['moe_monitor.py', 'inspect', '--model-dir', str(self.source)],
            ['moe_monitor.py', 'router-only', '--model-dir', str(self.source), '--output', str(report), '--keep-experts', '2', '--new-topk', '1', '--samples', '16'],
            ['moe_prune.py', 'dry-run', '--model-dir', str(self.source), '--plan', str(plan)],
            ['moe_prune.py', 'prune-checkpoint', '--model-dir', str(self.source), '--plan', str(plan), '--output-dir', str(self.base / 'cli-pruned')],
        ]
        for args in commands:
            result = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(result.returncode, 0, 'CLI stage failed: ' + args[1])
        self.assertEqual(json.loads(report.read_text())['mode'], 'router-only')
        self.assertFalse(self.source.with_name('source-dryrun').exists())
        self.assertEqual(json.loads((self.base / 'cli-pruned' / 'config.json').read_text())['num_local_experts'], 2)

if __name__ == '__main__': unittest.main()
