"""Offline tests. Synthetic values exist only inside temporary test directories."""
import json,tempfile,unittest
from pathlib import Path
import judge_run as j

class EvaluationTests(unittest.TestCase):
 def test_scores_reject_invalid_and_na_zero(self):
  good={'eh_score':2,'eh_reason':'Supported','text_present':False,'tr_score':None,'tr_reason':'No readable text'}
  self.assertEqual(j.validate_reply(good),good)
  for key,value in [('eh_score',True),('eh_score',3),('tr_score',0),('text_present','false')]:
   with self.assertRaises(ValueError):j.validate_reply({**good,key:value})
 def test_spearman_and_ties(self):
  self.assertAlmostEqual(j.spearman([0,1,2],[2,1,0]),-1)
  self.assertAlmostEqual(j.spearman([0,0,1,2],[0,0,1,2]),1)
  self.assertIsNone(j.spearman([1,1,1],[0,1,2]))
 def test_balanced_start_and_complete_order(self):
  rows=j.ordered(j.load_inputs())
  self.assertEqual(len({r['caption_id'] for r in rows}),500)
  self.assertEqual(set(r['stratum'] for r in rows[:8]),{'clear','blurry','low_light','text'})
  self.assertEqual(len({r['file_name'] for r in rows[:8]}),4)
 def test_incomplete_and_duplicate_results(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);(p/'judge_results.jsonl').write_text('{broken')
   with self.assertRaises(ValueError):j.read_results(p)
   (p/'judge_results.jsonl').unlink()
   self.assertEqual(j.read_results(p),[])
   with self.assertRaises(RuntimeError):j.analyze(p)

if __name__=="__main__":unittest.main()
