import sys
import unittest
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import download_historical_odds as dho


class DownloadHistoricalOddsTests(unittest.TestCase):
    def test_parse_month_valid(self) -> None:
        self.assertEqual(dho._parse_month("2023-05"), (2023, 5))

    def test_parse_month_invalid(self) -> None:
        with self.assertRaises(ValueError):
            dho._parse_month("2023")
        with self.assertRaises(ValueError):
            dho._parse_month("2023-13")

    def test_month_start_end_leap_year(self) -> None:
        tz = ZoneInfo("UTC")
        start, end = dho._month_start_end(2024, 2, tz)
        self.assertEqual(start.day, 1)
        self.assertEqual(end.day, 29)

    def test_iter_months_across_year(self) -> None:
        months = dho._iter_months(2023, 11, 2024, 2)
        self.assertEqual(months, [(2023, 11), (2023, 12), (2024, 1), (2024, 2)])


if __name__ == "__main__":
    unittest.main()
