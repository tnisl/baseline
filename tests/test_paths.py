from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from umilife.paths import find_data_root


class Paths(unittest.TestCase):
    def test_t4_does_not_select_emulated_bfloat16(self):
        from types import SimpleNamespace
        from umilife.attention import native_bf16_supported
        cuda=SimpleNamespace(is_available=lambda:True,
            get_device_capability=lambda:(7,5),is_bf16_supported=lambda:True)
        self.assertFalse(native_bf16_supported(SimpleNamespace(cuda=cuda)))
        cuda.get_device_capability=lambda:(8,0)
        self.assertTrue(native_bf16_supported(SimpleNamespace(cuda=cuda)))

    def test_finds_wrapped_dataset_without_entering_audio_or_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            data=root/"datasets/owner/dataset/trainpublic_test/train+public_test"
            for split in ("train","public_test"):
                (data/split).mkdir(parents=True)
                (data/split/f"{split}.jsonl").write_text("")
                (data/split/"audio").mkdir()
                (data/split/f"{split}_environment").mkdir()
            original=Path.iterdir
            def guarded(path):
                if path.name in ("audio","train_environment","public_test_environment"):
                    raise AssertionError("Data discovery entered sample data")
                return original(path)
            with patch.object(Path,"iterdir",guarded):
                found,archive=find_data_root(root)
            self.assertEqual(found,data)
            self.assertIsNone(archive)

    def test_returns_archive_and_missing_input_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaises(FileNotFoundError):
                find_data_root(root)
            (root/"train+public_test.zip").write_bytes(b"")
            found,archive=find_data_root(root)
            self.assertIsNone(found)
            self.assertEqual(archive,root/"train+public_test.zip")


if __name__=="__main__":
    unittest.main()
