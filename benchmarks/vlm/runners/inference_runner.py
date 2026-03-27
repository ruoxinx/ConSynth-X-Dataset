"""
Main inference runner with checkpoint/resume support.
Orchestrates: dataset loading → model inference → result saving.
"""
import json
import os
import time
import glob
import pandas as pd
from tqdm import tqdm


class InferenceRunner:
    """
    Runs inference for a given model + task + dataset condition.

    Usage:
        runner = InferenceRunner(
            model=model_instance,
            task=task_instance,
            condition='original',
            output_dir='output/vqa_safety',
        )
        runner.run()
    """

    def __init__(self, model, task, condition, output_dir,
                 tsv_path=None, checkpoint_every=1, resume=True):
        self.model = model
        self.task = task
        self.condition = condition
        self.output_dir = output_dir
        self.tsv_path = tsv_path
        self.checkpoint_every = checkpoint_every
        self.resume = resume

        os.makedirs(output_dir, exist_ok=True)

        # File paths
        model_name = getattr(model, 'model_id', 'unknown').split('/')[-1]
        self.out_file = os.path.join(
            output_dir,
            f'{model_name}_{task.task_name}_{condition}.json'
        )
        self.checkpoint_file = self.out_file.replace('.json', '_checkpoint.json')

    @staticmethod
    def _fix_path(path):
        """Fix known path mismatches in TSV files.
        TSVs use 'construction-site' but actual dir is 'ConstructionSite'.
        """
        return path.replace(
            '/VietHuy/construction-site/',
            '/VietHuy/ConstructionSite/'
        )

    def _load_samples(self):
        """Load samples from TSV or by scanning image directory."""
        if self.tsv_path and os.path.exists(self.tsv_path):
            samples = self.task.load_samples_from_tsv(self.tsv_path)
            for s in samples:
                s['image_path'] = self._fix_path(s['image_path'])
            return samples
        raise FileNotFoundError(f"TSV not found: {self.tsv_path}")

    def _find_weather_image(self, image_path):
        """For weather condition, find image across style directories."""
        if os.path.exists(image_path):
            return image_path
        # Try searching across weather style dirs
        img_name = os.path.basename(image_path)
        base_dir = os.path.dirname(os.path.dirname(image_path))
        candidates = glob.glob(os.path.join(base_dir, '*', 'images', img_name))
        if candidates:
            return candidates[0]
        return None

    def _load_checkpoint(self):
        """Load existing checkpoint results."""
        results = {}
        if self.resume and os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file) as f:
                results = json.load(f)
            print(f'Resumed from checkpoint: {len(results)} images done')
        return results

    def _save_checkpoint(self, results):
        """Save checkpoint."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def _save_final(self, results):
        """Save final sorted results and clean up checkpoint."""
        sorted_results = dict(sorted(results.items()))
        with open(self.out_file, 'w') as f:
            json.dump(sorted_results, f, indent=2, ensure_ascii=False)
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
        print(f'Saved: {self.out_file}')

    def run(self):
        """Run inference on all samples."""
        samples = self._load_samples()
        results = self._load_checkpoint()
        # Models that don't support few-shot (e.g. memory constraints)
        no_fewshot_models = {'gemma-3-27b-it'}
        # Models that need reduced few-shot count (OOM with full 5-shot)
        reduced_fewshot_models = {'Phi-4-multimodal-instruct': 2}
        model_name = getattr(self.model, 'model_id', '').split('/')[-1]
        uses_fewshot = self.task.uses_fewshot() and model_name not in no_fewshot_models

        # Prepare few-shot if needed
        few_shot_examples = []
        fewshot_intro = None
        if uses_fewshot:
            few_shot_examples = self.task.get_few_shot_examples()
            fewshot_intro = self.task.get_few_shot_intro()
            # Reduce few-shot count for memory-constrained models
            max_fs = reduced_fewshot_models.get(model_name)
            if max_fs and len(few_shot_examples) > max_fs:
                few_shot_examples = few_shot_examples[:max_fs]

        system_prompt = self.task.get_system_prompt()

        print(f'Task: {self.task.task_name}')
        print(f'Condition: {self.condition}')
        print(f'Samples: {len(samples)}')
        print(f'Already done: {len(results)}')
        print(f'Few-shot: {uses_fewshot} ({len(few_shot_examples)} examples)')

        t0 = time.time()
        done_count = 0

        for i, sample in enumerate(tqdm(samples, desc='Inference')):
            img_id = sample['image_id']
            if img_id in results:
                continue

            # Resolve image path
            image_path = sample['image_path']
            if self.condition == 'weather':
                resolved = self._find_weather_image(image_path)
                if resolved is None:
                    print(f'WARNING: Image not found: {image_path}')
                    results[img_id] = {'error': 'image not found'}
                    continue
                image_path = resolved
            elif not os.path.exists(image_path):
                print(f'WARNING: Image not found: {image_path}')
                results[img_id] = {'error': 'image not found'}
                continue

            prompt = self.task.get_user_prompt(sample)

            try:
                if uses_fewshot and few_shot_examples:
                    response = self.model.generate_with_fewshot(
                        few_shot_examples=few_shot_examples,
                        image=image_path,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        fewshot_intro=fewshot_intro,
                    )
                else:
                    response = self.model.generate(
                        image=image_path,
                        prompt=prompt,
                        system_prompt=system_prompt,
                    )

                parsed = self.task.parse_response(response)
                results[img_id] = {
                    'raw': response,
                    'parsed': parsed,
                }
            except Exception as e:
                print(f'ERROR [{img_id}]: {e}')
                results[img_id] = {'error': str(e)}

            done_count += 1

            # Checkpoint
            if done_count % self.checkpoint_every == 0:
                self._save_checkpoint(results)

        self._save_final(results)
        elapsed = time.time() - t0
        print(f'Done: {len(results)} images in {elapsed / 60:.1f} min')
        return results
