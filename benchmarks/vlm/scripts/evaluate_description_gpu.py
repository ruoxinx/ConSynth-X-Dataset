#!/usr/bin/env python3
"""
GPU evaluation script for description task.
Computes: BERTScore, BLEU, ROUGE-L, METEOR, CLIPScore on all prediction files.

Usage:
    python scripts/evaluate_description_gpu.py
"""
import json
import os
import sys
import glob
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / 'output' / 'description'
ANNOTATION_ROOT = Path(os.environ.get('CONSYNTH_ANNOTATION_ROOT',
                                       str(Path(os.environ.get('CONSYNTH_DATA_ROOT', Path.home() / 'consynth_data')).parent / 'ConstructionSite-10k-Implementation' / 'Annotations')))
REF_PATH = ANNOTATION_ROOT / 'random1_references.json'
LMUDATA = os.path.expanduser('~/LMUData')

# TSV paths per condition (to resolve image paths)
CONDITION_TSV = {
    'original': os.path.join(LMUDATA, 'CSAug_original_500.tsv'),
    'weather': os.path.join(LMUDATA, 'CSAug_weather_500.tsv'),
    'night': os.path.join(LMUDATA, 'CSAug_night_500.tsv'),
    'small': os.path.join(LMUDATA, 'CSAug_small_500.tsv'),
}


def build_image_id_to_path(condition):
    """Build mapping from image_id (e.g. '0000005') to image file path for a condition."""
    tsv_path = CONDITION_TSV.get(condition)
    if not tsv_path or not os.path.exists(tsv_path):
        print(f'  WARNING: TSV not found for condition {condition}: {tsv_path}')
        return {}
    df = pd.read_csv(tsv_path, sep='\t')
    mapping = {}
    for _, row in df.iterrows():
        img_path = row['image_path']
        img_id = os.path.basename(img_path).replace('.jpg', '').replace('.png', '')
        mapping[img_id] = img_path
    return mapping


def _find_weather_image(image_path):
    """For weather condition, find image across style directories."""
    if os.path.exists(image_path):
        return image_path
    img_name = os.path.basename(image_path)
    base_dir = os.path.dirname(os.path.dirname(image_path))
    candidates = glob.glob(os.path.join(base_dir, '*', 'images', img_name))
    if candidates:
        return candidates[0]
    return None


def align_predictions(predictions, references):
    """Align predictions with references by image_id."""
    pred_list, ref_list, ids = [], [], []
    for img_id, pred_data in predictions.items():
        if img_id not in references:
            continue
        pred_text = pred_data.get('parsed', pred_data.get('raw', ''))
        if isinstance(pred_text, dict):
            pred_text = pred_text.get('raw', str(pred_text))
        ref_text = references[img_id]
        if isinstance(ref_text, list):
            ref_text = ref_text[0] if ref_text else ''
        pred_list.append(str(pred_text))
        ref_list.append(str(ref_text))
        ids.append(img_id)
    return pred_list, ref_list, ids


def count_errors(predictions):
    """Count samples with errors or empty predictions."""
    errors = 0
    for img_id, pred_data in predictions.items():
        raw = pred_data.get('raw', '')
        if not raw or pred_data.get('error'):
            errors += 1
    return errors


def compute_bleu(preds, refs):
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    refs_tok = [[r.split()] for r in refs]
    hyps_tok = [p.split() for p in preds]
    smooth = SmoothingFunction().method1
    return corpus_bleu(refs_tok, hyps_tok, smoothing_function=smooth)


def compute_rouge_l(preds, refs):
    from rouge import Rouge
    rouge = Rouge()
    scores = rouge.get_scores(preds, refs, avg=True)
    return scores['rouge-l']['f']


def compute_meteor(preds, refs):
    import nltk
    from nltk.translate.meteor_score import meteor_score
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    scores = [meteor_score([r.split()], p.split()) for p, r in zip(preds, refs)]
    return sum(scores) / len(scores)


def compute_bertscore(preds, refs):
    from bert_score import BERTScorer
    # Match paper: RoBERTa-large + rescale_with_baseline=True
    scorer = BERTScorer(lang='en', rescale_with_baseline=True)
    P, R, F1 = scorer.score(preds, refs)
    return {
        'bertscore_p': P.mean().item(),
        'bertscore_r': R.mean().item(),
        'bertscore_f1': F1.mean().item(),
    }


def compute_sbert_similarity(preds, refs):
    """
    Compute Sentence-BERT cosine similarity between predictions and references.
    Length-invariant: encodes entire sentences into fixed-size vectors.
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    model = SentenceTransformer('all-MiniLM-L6-v2')
    pred_embs = model.encode(preds, batch_size=64, show_progress_bar=False)
    ref_embs = model.encode(refs, batch_size=64, show_progress_bar=False)

    # Per-pair cosine similarity
    scores = [float(cos_sim([p], [r])[0][0]) for p, r in zip(pred_embs, ref_embs)]
    return float(np.mean(scores))


def compute_clipscore(preds, ids, id_to_path, condition):
    """
    Compute CLIPScore: cosine similarity between CLIP image and text embeddings.
    Uses openai-clip (ViT-B/32) matching the author's approach.
    """
    import torch
    import clip
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Resolve image paths for aligned ids
    image_paths = []
    valid_preds = []
    for img_id, pred_text in zip(ids, preds):
        img_path = id_to_path.get(img_id)
        if not img_path:
            continue
        if condition == 'weather':
            img_path = _find_weather_image(img_path)
        if img_path and os.path.exists(img_path):
            image_paths.append(img_path)
            valid_preds.append(pred_text)

    if not image_paths:
        print(f'  WARNING: No images found for CLIPScore')
        return None

    print(f'  CLIPScore: computing on {len(image_paths)} images...')

    # Compute in batches
    batch_size = 32
    all_scores = []

    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_texts = valid_preds[i:i + batch_size]

            # Encode images
            images = []
            for p in batch_paths:
                try:
                    img = preprocess(Image.open(p).convert('RGB'))
                    images.append(img)
                except Exception as e:
                    print(f'  WARNING: Failed to load {p}: {e}')
                    images.append(preprocess(Image.new('RGB', (224, 224))))
            images = torch.stack(images).to(device)
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Encode texts (truncate to 77 tokens - CLIP limit)
            text_tokens = clip.tokenize(batch_texts, truncate=True).to(device)
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Cosine similarity per pair
            scores = (image_features * text_features).sum(dim=-1).cpu().numpy()
            all_scores.extend(scores.tolist())

    # CLIPScore = max(100 * cos_sim, 0) following standard convention
    clipscore = float(np.mean([max(2.5 * s, 0) for s in all_scores]))
    return clipscore


def main():
    print(f'Reference: {REF_PATH}')
    with open(REF_PATH) as f:
        references = json.load(f)
    print(f'Loaded {len(references)} references')

    pred_files = sorted(glob.glob(str(OUTPUT_DIR / '*.json')))
    # Skip checkpoint files (incomplete runs)
    pred_files = [f for f in pred_files if 'checkpoint' not in os.path.basename(f)]
    print(f'Found {len(pred_files)} prediction files')

    if not pred_files:
        print('No prediction files found. Exiting.')
        return

    all_results = []

    for pred_file in pred_files:
        fname = os.path.basename(pred_file)
        print(f'\n{"="*60}')
        print(f'Evaluating: {fname}')
        print(f'{"="*60}')

        with open(pred_file) as f:
            predictions = json.load(f)

        errors = count_errors(predictions)
        preds, refs, ids = align_predictions(predictions, references)
        print(f'  Samples: {len(predictions)}, Aligned: {len(preds)}, Errors: {errors}')

        if len(preds) == 0:
            print('  SKIPPED: no aligned samples')
            continue

        # Parse model name and condition from filename
        # Format: ModelName_description_condition.json
        base = fname.replace('.json', '')
        parts = base.split('_description_')
        model_name = parts[0] if len(parts) > 1 else base
        condition = parts[1] if len(parts) > 1 else 'unknown'

        # Compute text metrics
        bleu = compute_bleu(preds, refs)
        rouge_l = compute_rouge_l(preds, refs)
        meteor = compute_meteor(preds, refs)
        bert_scores = compute_bertscore(preds, refs)

        # Compute Sentence-BERT similarity (length-invariant semantic similarity)
        sbert_sim = compute_sbert_similarity(preds, refs)

        # Compute CLIPScore (needs images)
        id_to_path = build_image_id_to_path(condition)
        clipscore = compute_clipscore(preds, ids, id_to_path, condition)

        result = {
            'model': model_name,
            'condition': condition,
            'file': fname,
            'n_samples': len(preds),
            'n_errors': errors,
            'bleu': bleu,
            'rouge_l': rouge_l,
            'meteor': meteor,
            **bert_scores,
            'sbert_similarity': sbert_sim,
            'clipscore': clipscore,
        }
        all_results.append(result)

        print(f'  BLEU:       {bleu:.4f}')
        print(f'  ROUGE-L:    {rouge_l:.4f}')
        print(f'  METEOR:     {meteor:.4f}')
        print(f'  BERTScore:  P={bert_scores["bertscore_p"]:.4f}  '
              f'R={bert_scores["bertscore_r"]:.4f}  '
              f'F1={bert_scores["bertscore_f1"]:.4f}')
        print(f'  SBERT-Sim:  {sbert_sim:.4f}')
        print(f'  CLIPScore:  {clipscore:.4f}' if clipscore is not None else '  CLIPScore:  N/A')

    # Save results
    results_path = PROJECT_ROOT / 'output' / 'description_eval_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n{"="*60}')
    print(f'Saved results: {results_path}')
    print(f'Evaluated {len(all_results)} files total.')

    # Print summary table
    print(f'\n{"="*60}')
    print(f'{"Model":<30} {"Cond":<10} {"BLEU":>8} {"ROUGE-L":>8} {"METEOR":>8} {"BERT-F1":>8} {"SBERT":>8} {"CLIP":>8}')
    print('-' * 100)
    for r in all_results:
        clip_str = f'{r["clipscore"]:>8.4f}' if r.get('clipscore') is not None else '     N/A'
        print(f'{r["model"]:<30} {r["condition"]:<10} '
              f'{r["bleu"]:>8.4f} {r["rouge_l"]:>8.4f} '
              f'{r["meteor"]:>8.4f} {r["bertscore_f1"]:>8.4f} '
              f'{r["sbert_similarity"]:>8.4f} {clip_str}')


if __name__ == '__main__':
    main()
