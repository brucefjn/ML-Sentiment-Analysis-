import hashlib
from pathlib import Path
import tempfile
import unittest

from data_utils import validate_csv


class InputValidation(unittest.TestCase):
    def check(self, text, expected=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            path.write_text(text)
            return validate_csv(path, expected)

    def test_duplicates_are_reported_not_silently_removed(self):
        report = self.check("Review,Emotion\ngood,positive\ngood,positive\nbad,negative\n")
        self.assertEqual(report["extra_duplicate_rows"], 1)
        self.assertEqual(report["rows"], 3)

    def test_checksum_rejects_changed_data(self):
        digest = hashlib.sha256(b"original bytes").hexdigest()
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.check("Review,Emotion\ngood,positive\n", digest)

    def test_download_error_page_is_not_accepted(self):
        with self.assertRaisesRegex(ValueError, "columns"):
            self.check("<html>Sign in</html>")

    def test_empty_review_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing review"):
            self.check("Review,Emotion\n   ,positive\n")

    def test_unknown_label_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "label"):
            self.check("Review,Emotion\ngood,neutral\n")

    def test_conflicting_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.check("Review,Emotion\ngood,positive\ngood,negative\n")


if __name__ == "__main__":
    unittest.main()
