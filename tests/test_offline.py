"""Offline regression checks with synthetic data, not research measurements."""
import ast
import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

class OfflineTests(unittest.TestCase):
    def test_sources_parse(self):
        for path in (ROOT / 'src').glob('*.py'):
            with self.subTest(path=path.name):
                ast.parse(path.read_text(encoding='utf-8'))

    def test_train_apply_and_threshold_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            rng = np.random.default_rng(23)
            for name in ['dev', 'pos', 'neg', 'eval']:
                (work/name).mkdir()
            for i in range(12):
                name = f'record{i}.csv'
                data = np.column_stack([np.zeros((4,4)), rng.uniform(1,10,(4,8))])
                for folder in ['dev', 'neg' if i<8 else 'pos']:
                    np.savetxt(work/folder/name, data, delimiter=',', header=','.join(f'c{j}' for j in range(12)), comments='')
                if i<3:
                    np.savetxt(work/'eval'/name, data, delimiter=',', header=','.join(f'c{j}' for j in range(12)), comments='')
            def run(script, *args):
                result = subprocess.run([sys.executable,str(ROOT/'src'/script),*args], cwd=work,
                    env={**os.environ,'MPLBACKEND':'Agg','PYTHONIOENCODING':'utf-8'}, capture_output=True,text=True,encoding='utf-8')
                self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            run('train_iforest.py','--dev_dir','dev','--dev_posdir','pos','--dev_negdir','neg','--pca_var','0.95','--n_estimators','10')
            run('apply_iforest.py','--model','iforest_model.joblib','--eval_dir','eval')
            with (work/'eval_predictions_if.csv').open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),3)
            self.assertTrue(all(row['pred_label'] in ['0','1'] for row in rows))
            for name,values in [('dev_neg_features.csv',[1,2,3,4]),('dev_pos_features.csv',[6,7,8,9])]:
                (work/name).write_text('spectral_flatness\n'+'\n'.join(map(str,values))+'\n')
            run('threshold_optimisation.py')
            with (work/'dev_thresholds_summary.csv').open() as f:
                result=list(csv.DictReader(f))
            self.assertEqual(len(result),1)
            self.assertEqual(float(result[0]['accuracy_dev']),1.0)

if __name__ == '__main__':
    unittest.main()
