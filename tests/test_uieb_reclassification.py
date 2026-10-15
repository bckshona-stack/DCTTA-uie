import csv
import importlib.util
import json
from pathlib import Path
import unittest
import uuid
from contextlib import contextmanager, redirect_stdout
import io

from PIL import Image


SPEC = importlib.util.spec_from_file_location('reclassify_uieb', Path(__file__).parents[1] / 'reclassify_uieb.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@contextmanager
def scratch():
    # Retain small test fixtures; repository policy prohibits bulk deletion.
    path = Path(__file__).parents[1] / 'tmp' / 'uieb_copy_tests' / uuid.uuid4().hex
    path.mkdir(parents=True)
    with redirect_stdout(io.StringIO()):
        yield path


class DatasetCopyTests(unittest.TestCase):
    def setup_case(self, path):
        source = path / 'original'
        for role in ('raw', 'reference'):
            (source / f'{role}-890').mkdir(parents=True)
        rows = []
        for name, size, mode, color in [('odd_name.png', (17, 23), 'RGBA', (32, 120, 180, 77)),
                                         ('other.jpg', (31, 19), 'RGB', (220, 83, 47))]:
            for role in ('raw', 'reference'):
                Image.new(mode, size, color).save(source / f'{role}-890' / name)
            rows.append(dict(sample_id=name, primary_category='blue_tint', severity='light',
                             blue_level=1, green_level=0, haze_level=0, lowlight_level=0,
                             secondary_categories='', review_note='synthetic copy test', scene_group_id='',
                             raw_relative=f'raw/light/blue_tint/{name}',
                             reference_relative=f'reference/light/blue_tint/{name}',
                             width=size[0], height=size[1],
                             raw_sha256=MODULE.sha256(source / 'raw-890' / name),
                             reference_sha256=MODULE.sha256(source / 'reference-890' / name)))
        labels = path / 'labels.csv'
        labels.write_bytes(MODULE.csv_bytes(rows, list(rows[0])))
        eligibility = path / 'eligibility.json'
        eligibility.write_text(json.dumps(dict(eligible=[dict(sample_id=r['sample_id']) for r in rows],
                                              original_count=2, excluded_count=0, excluded=[])))
        return source, labels, eligibility, path / 'copied', rows

    def test_exact_copy_odd_dimensions_alpha_jpeg_and_resume(self):
        with scratch() as temporary:
            source, labels, eligibility, output, rows = self.setup_case(Path(temporary))
            MODULE.materialize(source, labels, eligibility, output)
            for row in rows:
                for role in ('raw', 'reference'):
                    self.assertEqual((source / f'{role}-890' / row['sample_id']).read_bytes(),
                                     (output / row[f'{role}_relative']).read_bytes())
            MODULE.materialize(source, labels, eligibility, output)
            MODULE.materialize(source, labels, eligibility, output, verify_only=True)

    def test_corrupt_existing_image_is_rejected_and_never_overwritten(self):
        with scratch() as temporary:
            source, labels, eligibility, output, rows = self.setup_case(Path(temporary))
            MODULE.materialize(source, labels, eligibility, output)
            bad = output / rows[0]['raw_relative']
            bad.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'Refusing to overwrite'):
                MODULE.materialize(source, labels, eligibility, output)
            self.assertEqual(bad.read_bytes(), b'corrupt')

    def test_source_change_and_cross_end_near_duplicate_are_rejected(self):
        with scratch() as temporary:
            source, labels, eligibility, output, rows = self.setup_case(Path(temporary))
            (source / 'raw-890' / rows[0]['sample_id']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Source file changed'):
                MODULE.materialize(source, labels, eligibility, output)
            self.assertFalse(output.exists())
            rows[0]['scene_group_id'] = rows[1]['scene_group_id'] = 'same_scene'
            rows[1].update(severity='severe', blue_level=3,
                           raw_relative='raw/severe/blue_tint/other.jpg',
                           reference_relative='reference/severe/blue_tint/other.jpg')
            with self.assertRaisesRegex(ValueError, 'crosses light/severe'):
                MODULE.validate_rows(rows, dict(eligible=[dict(sample_id=r['sample_id']) for r in rows]))


if __name__ == '__main__':
    unittest.main()
