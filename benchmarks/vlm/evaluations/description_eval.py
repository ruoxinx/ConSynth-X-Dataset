"""
Evaluation for image captioning / description generation.
Metrics: BERTScore, BLEU, ROUGE-L, METEOR, CIDEr, CLIPScore.
"""
import json


class DescriptionEvaluator:
    """Evaluates description generation quality."""

    def __init__(self, predictions_path, references_path):
        with open(predictions_path) as f:
            self.predictions = json.load(f)
        with open(references_path) as f:
            self.references = json.load(f)

    def compute_bertscore(self):
        """Compute BERTScore (P, R, F1)."""
        from bert_score import score as bert_score

        pred_list, ref_list = self._align()
        P, R, F1 = bert_score(pred_list, ref_list, lang='en', verbose=True)
        return {
            'precision': P.mean().item(),
            'recall': R.mean().item(),
            'f1': F1.mean().item(),
        }

    def compute_bleu(self):
        """Compute BLEU score."""
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

        pred_list, ref_list = self._align()
        refs = [[r.split()] for r in ref_list]
        hyps = [p.split() for p in pred_list]
        smooth = SmoothingFunction().method1
        score = corpus_bleu(refs, hyps, smoothing_function=smooth)
        return {'bleu': score}

    def compute_rouge(self):
        """Compute ROUGE-L score."""
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        pred_list, ref_list = self._align()
        scores = [scorer.score(r, p)['rougeL'].fmeasure
                  for p, r in zip(pred_list, ref_list)]
        return {'rouge_l': sum(scores) / len(scores)}

    def compute_meteor(self):
        """Compute METEOR score."""
        import nltk
        from nltk.translate.meteor_score import meteor_score

        pred_list, ref_list = self._align()
        scores = [meteor_score([r.split()], p.split())
                  for p, r in zip(pred_list, ref_list)]
        return {'meteor': sum(scores) / len(scores)}

    def compute_clipscore(self, image_dir=None):
        """Compute CLIPScore (requires images)."""
        # Placeholder - CLIPScore needs actual images
        return {'clipscore': None, 'note': 'requires image_dir'}

    def compute_all(self):
        """Compute all text-based metrics."""
        results = {}
        results.update(self.compute_bertscore())
        results.update(self.compute_bleu())
        results.update(self.compute_rouge())
        results.update(self.compute_meteor())
        return results

    def _align(self):
        """Align predictions with references by image_id."""
        pred_list = []
        ref_list = []
        for img_id, pred_data in self.predictions.items():
            if img_id not in self.references:
                continue
            pred_text = pred_data.get('parsed', pred_data.get('raw', ''))
            if isinstance(pred_text, dict):
                pred_text = pred_text.get('raw', str(pred_text))
            ref_text = self.references[img_id]
            if isinstance(ref_text, list):
                ref_text = ref_text[0] if ref_text else ''
            pred_list.append(str(pred_text))
            ref_list.append(str(ref_text))
        return pred_list, ref_list
