"""Checks for relevance accounting, stable exact search and streaming dataset loading."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from performance_baseline import DiskBatches, exact_neighbors, load_dataset, ranking_metrics


class BaselineTests(unittest.TestCase):
    def test_relevance_cutoffs(self):
        ids = list(range(100))
        result = ranking_metrics(ids, {0: 1, 10: 1, 100: 1})
        self.assertEqual(result['recall_at_10'], 1 / 3)
        self.assertEqual(result['recall_at_100'], 2 / 3)
        self.assertEqual(result['mrr_at_10'], 1)
        self.assertAlmostEqual(result['ndcg_at_10'], 1 / (1 + 1 / math.log2(3) + 0.5))
        self.assertTrue(all(value == 0 for value in ranking_metrics([], {0: 1}).values()))
        with self.assertRaises(ValueError):
            ranking_metrics([0, 0], {0: 1})

    def test_blocked_cosine_matches_exhaustive_including_ties(self):
        rng = np.random.default_rng(7)
        vectors = rng.normal(size=(31, 8)).astype(np.float32)
        vectors[19] = vectors[0]
        queries = vectors[[0, 1, 2]] * 7
        v, q = vectors.astype(np.float64), queries.astype(np.float64)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        expected = np.argsort(-(q @ v.T), axis=1, kind='stable')[:, :10].tolist()
        for block in (1, 7, 64):
            self.assertEqual(exact_neighbors(vectors, queries, 10, block), expected)
        with self.assertRaises(ValueError):
            exact_neighbors(np.zeros((2, 8)), queries, 1)

    def test_lexical_dataset_uses_dev_labels_and_streams(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'corpus.jsonl').write_text('\n'.join(json.dumps(d) for d in [
                {'_id': 'negative', 'title': '标题', 'text': '干扰文档'},
                {'_id': 'positive', 'title': '標題', 'text': '繼承者們 Rust API 軟體'}]))
            (root / 'queries.jsonl').write_text(json.dumps({'_id': 'q', 'text': '繼承者們'}))
            (root / 'dev.tsv').write_text('query-id\tcorpus-id\tscore\nq\tnegative\t0\nq\tpositive\t1\n')
            manifest = {'documents': 2, 'dimensions': 0, 'qrels_file': 'dev.tsv',
                        'tokenizer': 'UnicodeAlphanumericZH', 'files_sha256': {
                            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            args = argparse.Namespace(dataset=directory, modes=['lexical'], top_k=1, batch_size=1)
            batches, payloads, truth, metadata = load_dataset(args)
            try:
                self.assertIsInstance(batches, DiskBatches)
                self.assertIsNone(truth)
                self.assertEqual(metadata['_qrels'], [{1: 1.0}])
                self.assertEqual(metadata['query_ids'], ['q'])
                self.assertEqual(len(list(batches)), 2)
                self.assertEqual(len(list(batches)), 2)  # iterator is rewindable
                self.assertNotIn('vector', json.loads(next(iter(batches)))[0])
                self.assertEqual(json.loads(payloads['lexical'][0])['operator'], 'or')
            finally:
                batches.close()
            for mode, expected_body, expected_query in [
                    ('identity', '標題\n繼承者們 Rust API 軟體', '繼承者們'),
                    ('t2s', '标题\n继承者们 Rust API 软体', '继承者们')]:
                args.text_normalization = mode
                batches, payloads, _, metadata = load_dataset(args)
                try:
                    document = json.loads(list(batches)[1])[0]
                    self.assertEqual(document['record_id'], 1)
                    self.assertEqual(document['body'], '標題\n繼承者們 Rust API 軟體')
                    self.assertEqual(document['search_body'], expected_body)
                    self.assertEqual(json.loads(payloads['lexical'][0])['query'], expected_query)
                    self.assertEqual(metadata['_qrels'], [{1: 1.0}])
                    stats = metadata['text_normalization']
                    self.assertEqual(stats['documents']['count'], 2)
                    self.assertEqual(stats['queries']['count'], 1)
                    self.assertEqual(stats['documents']['changed'], int(mode == 't2s'))
                    self.assertEqual(stats['queries']['changed'], int(mode == 't2s'))
                finally:
                    batches.close()
            args.modes = ['hybrid']
            with self.assertRaisesRegex(ValueError, 'lexical-only'):
                load_dataset(args)
            args.modes = ['lexical']
            (root / 'corpus.jsonl').write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_dataset(args)


if __name__ == '__main__':
    unittest.main()
