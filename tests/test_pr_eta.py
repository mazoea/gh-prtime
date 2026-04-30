import sys
import os
import unittest

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_this_dir, ".."))


class TestParseEta(unittest.TestCase):

    def testparse(self):
        """ Smoke test: a well-formed ETA table parses to a non-None table. """
        from prtime import parse_eta_body
        s = """
| Phases            | JH  |  JP  | TM |   JM | Total  |
|-----------------|----:|----:|-----:|-----:|-------:|
| ETA                  |  0  |    |     3 |      0 |        3 |
| Developing      |  5+4,5+8+8+5+8+5  |    0,5 + 1 |    0 |      0 |       43.5 |
| Review             |  4+4  |     2.5 + 0.5 + 1 + 1 |    0 |      0 |         13 |
| Total                |   -  |   -   |  -    |   -    |       59.5 |
| ETA est.             |      |       |       |         |     40  |
| ETA cust.           |   -  |   -  |   -   |   -     |        40 |
        """
        d = parse_eta_body(s, "test:1:smoke")
        self.assertIsNotNone(d)


class TestSafeSumHours(unittest.TestCase):
    """The cell-summing path used to call ``eval()`` on PR text, which is
    arbitrary code execution by anyone who can edit a tracked PR/issue body.
    These tests pin the safe-parser behavior."""

    def test_simple_int(self):
        from prtime import _safe_sum_hours
        self.assertEqual(_safe_sum_hours("5"), 5.0)

    def test_dot_decimal(self):
        from prtime import _safe_sum_hours
        self.assertAlmostEqual(_safe_sum_hours("2.5"), 2.5)

    def test_comma_decimal(self):
        from prtime import _safe_sum_hours
        self.assertAlmostEqual(_safe_sum_hours("4,5"), 4.5)

    def test_addition_mixed_separators(self):
        from prtime import _safe_sum_hours
        self.assertAlmostEqual(_safe_sum_hours("5+4,5+8+8+5+8+5"), 43.5)

    def test_addition_with_spaces(self):
        from prtime import _safe_sum_hours
        self.assertAlmostEqual(_safe_sum_hours("2.5 + 0.5 + 1 + 1"), 5.0)

    def test_subtraction(self):
        from prtime import _safe_sum_hours
        self.assertAlmostEqual(_safe_sum_hours("5-1.5"), 3.5)

    def test_empty_and_dash(self):
        from prtime import _safe_sum_hours
        self.assertEqual(_safe_sum_hours(""), 0.0)
        self.assertEqual(_safe_sum_hours("   "), 0.0)
        self.assertEqual(_safe_sum_hours("-"), 0.0)

    def test_rejects_code_injection(self):
        """Anything that previously made eval() exploitable must raise."""
        from prtime import _safe_sum_hours
        for evil in [
            "__import__('os').system('echo pwned')",
            "open('/etc/passwd').read()",
            "1+abs(1)",
            "1; print(1)",
            "1**2",
        ]:
            with self.assertRaises(ValueError, msg=f"should reject {evil!r}"):
                _safe_sum_hours(evil)

    def test_rejects_unicode_digits(self):
        """\\d would match Unicode-property digits like Arabic-Indic digits;
        the literal [0-9] allowlist must reject them."""
        from prtime import _safe_sum_hours
        # U+0660..U+0669 are Arabic-Indic digits; they're "digits" to \d.
        with self.assertRaises(ValueError):
            _safe_sum_hours("١+٢")

    def test_rejects_oversized_input(self):
        """Cap input length so a PR author can't DoS the parser."""
        from prtime import _safe_sum_hours
        long_expr = "1" + ("+1" * 1000)
        with self.assertRaises(ValueError):
            _safe_sum_hours(long_expr)


class TestPrevMonday(unittest.TestCase):
    """`prev_monday` was using local-time `datetime.today()` while the rest of
    the pipeline uses UTC PyGithub timestamps. Pin it to UTC."""

    def test_returns_a_monday(self):
        from prtime import prev_monday
        d = prev_monday()
        self.assertEqual(d.weekday(), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
