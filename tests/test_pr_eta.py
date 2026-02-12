import sys
import os
import unittest
from datetime import datetime, timedelta
from collections import defaultdict
from unittest.mock import MagicMock

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_this_dir, ".."))


def _build_week_year(created, closed):
    """
        Replicate the week iteration logic from process_one to test it.
    """
    start_iso = created.isocalendar()
    week_d_start, year_start = start_iso[1], start_iso[0]
    end_d = closed or datetime.now()
    end_iso = end_d.isocalendar()
    week_d_end, year_end = end_iso[1], end_iso[0]

    week_year = []
    d = created.date() if isinstance(created, datetime) else created
    d = d - timedelta(days=d.weekday())
    end_date = end_d.date() if isinstance(end_d, datetime) else end_d
    while d <= end_date:
        iso = d.isocalendar()
        week_year.append((iso[1], iso[0]))
        d += timedelta(weeks=1)
    return week_year


class Testbasic(unittest.TestCase):

    def testparse(self):
        """ testparse """
        from prtime import parse_eta_lines
        s = """
| Phases            | JH  |  JP  | TM |   JM | Total  |
|-----------------|----:|----:|-----:|-----:|-------:|
| ETA                  |  0  |    |     3 |      0 |        3 |
| Developing      |  5+4,5+8+8+5+8+5  |    0,5 + 1 |    0 |      0 |         45 |
| Review             |  4+4  |     2.5 + 0.5 + 1 + 1 |    0 |      0 |         13 |
| Total                |   -  |   -   |  -    |   -    |         61 |
| ETA est.             |      |       |       |         |     40  |
| ETA cust.           |   -  |   -  |   -   |   -     |        40 |
        """
        pr = MagicMock()
        pr.body = s
        lines, ignored = parse_eta_lines(pr)
        self.assertTrue(len(lines) > 0)
        self.assertFalse(ignored)


class TestWeekIteration(unittest.TestCase):

    def test_same_week(self):
        """PR created and closed in the same week."""
        created = datetime(2025, 9, 15)  # Monday, week 38
        closed = datetime(2025, 9, 17)   # Wednesday, week 38
        weeks = _build_week_year(created, closed)
        self.assertEqual(weeks, [(38, 2025)])

    def test_two_weeks_same_year(self):
        """PR spanning two weeks in the same year."""
        created = datetime(2025, 9, 15)  # week 38
        closed = datetime(2025, 9, 22)   # week 39
        weeks = _build_week_year(created, closed)
        self.assertEqual(weeks, [(38, 2025), (39, 2025)])

    def test_cross_year_boundary(self):
        """PR crossing a year boundary."""
        created = datetime(2024, 12, 23)  # week 52 of 2024
        closed = datetime(2025, 1, 8)     # week 2 of 2025
        weeks = _build_week_year(created, closed)
        # Should include week 52/2024, week 1/2025, week 2/2025
        self.assertEqual(len(weeks), 3)
        self.assertEqual(weeks[0], (52, 2024))
        self.assertEqual(weeks[-1], (2, 2025))
        # No week 0 should exist
        week_nums = [w for w, y in weeks]
        self.assertNotIn(0, week_nums)

    def test_year_with_week_53(self):
        """Test a year that has ISO week 53 (e.g., 2020)."""
        created = datetime(2020, 12, 28)  # week 53 of 2020
        closed = datetime(2021, 1, 5)     # week 1 of 2021
        weeks = _build_week_year(created, closed)
        self.assertIn((53, 2020), weeks)
        self.assertIn((1, 2021), weeks)
        week_nums = [w for w, y in weeks]
        self.assertNotIn(0, week_nums)

    def test_long_running_pr(self):
        """PR open for many weeks does not produce infinite loop."""
        created = datetime(2025, 1, 6)   # week 2
        closed = datetime(2025, 9, 15)   # week 38
        weeks = _build_week_year(created, closed)
        # Should have ~37 weeks
        self.assertGreater(len(weeks), 30)
        self.assertLess(len(weeks), 40)
        # All week numbers should be valid (1-53)
        for w, y in weeks:
            self.assertGreaterEqual(w, 1)
            self.assertLessEqual(w, 53)

    def test_week_52_end(self):
        """PR ending in week 52 should not cause issues."""
        created = datetime(2025, 12, 15)  # week 51
        closed = datetime(2025, 12, 22)   # week 52
        weeks = _build_week_year(created, closed)
        self.assertEqual(len(weeks), 2)
        week_nums = [w for w, y in weeks]
        self.assertIn(51, week_nums)
        self.assertIn(52, week_nums)


if __name__ == '__main__':
    # unittest.main()
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(Testbasic))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestWeekIteration))
    unittest.TextTestRunner(verbosity=2).run(suite)
