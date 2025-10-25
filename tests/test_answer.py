# paperqa/tests/test_answer.py
"""Tiny tests for AnswerTool (no external deps)."""

import unittest
from tools.tool_answer import AnswerTool, Evidence, Background

class TestAnswerTool(unittest.TestCase):
    def setUp(self):
        self.tool = AnswerTool(min_evidence=5, top_k=8, min_avg_score=6.5)
        self.question = "What is the mechanism of mRNA vaccine delivery?"
        self.bg = Background(question=self.question, background_text="mRNA vaccines use lipid nanoparticles to deliver mRNA into cells.")

    def _evidence(self, n_good=5, n_bad=2):
        ev = []
        # good evidence (scores around 8-9)
        for i in range(n_good):
            ev.append(Evidence(
                chunk_id=f"good#{i}",
                summary=f"LNPs protect mRNA and facilitate endocytosis leading to cytosolic release (detail {i}).",
                score=8.5,
                citation=f"(Doe 202{i})"
            ))
        # bad/irrelevant evidence (scores around 3-4)
        for j in range(n_bad):
            ev.append(Evidence(
                chunk_id=f"bad#{j}",
                summary="Unrelated observation about adjuvants.",
                score=3.5,
                citation=f"(Foo 201{j})"
            ))
        return ev

    def test_confident_answer(self):
        ev = self._evidence()
        ans = self.tool.generate(self.question, ev, self.bg)
        self.assertNotIn("cannot answer", ans.text.lower())
        self.assertGreaterEqual(ans.confidence, 0.6)
        self.assertTrue(any("(Doe" in c for c in ans.citations))

    def test_insufficient_evidence(self):
        ev = self._evidence(n_good=3, n_bad=0)  # fewer than min_evidence
        ans = self.tool.generate(self.question, ev, self.bg)
        self.assertIn("cannot answer", ans.text.lower())
        self.assertLess(ans.confidence, 0.1)
        self.assertEqual(ans.citations, [])

if __name__ == "__main__":
    unittest.main()
