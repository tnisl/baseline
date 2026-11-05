import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]


class NotebookTests(unittest.TestCase):
    def test_cells_and_embedded_source_match_repository(self):
        notebook = json.loads((ROOT / "notebooks/planner/kaggle_planner.ipynb").read_text())
        payload = None
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            ast.parse(cell["source"])
            self.assertEqual(cell["outputs"], [])
            self.assertIsNone(cell["execution_count"])
            if cell["source"].startswith("blob = base64.b64decode("):
                payload = ast.literal_eval(ast.parse(cell["source"]).body[0].value.args[0])
        self.assertIsNotNone(payload)
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
            manifest = json.loads(archive.read("source_manifest.json"))
            expected = {str(p.relative_to(ROOT)) for p in (ROOT / "src/twonode").glob("*.py")}
            self.assertEqual(set(manifest), expected | {"split.txt", "requirements.txt"})
            for name, digest in manifest.items():
                data = archive.read(name)
                self.assertEqual(hashlib.sha256(data).hexdigest(), digest)
                source = ROOT / "notebooks/planner/requirements.txt" if name == "requirements.txt" else ROOT / name
                self.assertEqual(data, source.read_bytes(), name)
            self.assertEqual(len(archive.read("split.txt").decode().splitlines()), 200)
            self.assertFalse(any("environment" in name for name in archive.namelist()))


if __name__ == "__main__":
    unittest.main()
