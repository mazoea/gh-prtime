import sys
import os
import unittest

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_this_dir, ".."))


class Testbasic(unittest.TestCase):

    def testparse(self):
        """ testparse """
        from prtime import parse_eta
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
        d = parse_eta(s)
        self.assertTrue(d is not None)


class TestSumHours(unittest.TestCase):
    """
    `sum_hours` parses arithmetic out of PR-body table cells, which are
    fully attacker-controlled. These tests pin both directions:
      * legitimate arithmetic still evaluates to the right float;
      * anything that is not plain numeric arithmetic (function calls,
        names, attribute access, dunder/sandbox-escape tricks, operators
        outside +-*/) is rejected and returns -1.0 -- i.e. no code runs.
    """

    def setUp(self):
        from prtime import sum_hours
        self.sum_hours = sum_hours

    def test_valid_arithmetic_parses(self):
        cases = {
            "1": 1.0,
            "0": 0.0,
            "1+2": 3.0,
            "1.5*2": 3.0,
            "(1+0.25)*4": 5.0,
            "(1+2)*3": 9.0,
            "5+4+8": 17.0,
            "2.5 + 0.5 + 1 + 1": 5.0,
            "-2": -2.0,
            "10/4": 2.5,
        }
        for expr, expected in cases.items():
            with self.subTest(expr=expr):
                self.assertEqual(self.sum_hours(expr, "test"), expected)

    def test_non_arithmetic_rejected(self):
        # Empty / None / non-numeric / wrong characters -> -1.0
        for expr in (None, "", "   ", "abc", "1,2", "5,5", "1\n2",
                     "os.system", "1 ; 2", "1 & 2"):
            with self.subTest(expr=expr):
                self.assertEqual(self.sum_hours(expr, "test"), -1.0)

    def test_no_code_execution(self):
        # Classic eval()-injection payloads must never execute; each must
        # be rejected (return -1.0) either by the character allowlist or by
        # the AST node allowlist.
        payloads = [
            '__import__("os").system("echo PWNED")',
            'open("/etc/passwd").read()',
            '().__class__.__bases__[0].__subclasses__()',
            'eval("1")',
            'exec("x=1")',
            'globals()',
            'lambda: 1',
            '[x for x in range(3)]',
        ]
        for expr in payloads:
            with self.subTest(expr=expr):
                self.assertEqual(self.sum_hours(expr, "test"), -1.0)

    def test_ast_disallowed_operators_rejected(self):
        # These pass the character allowlist (only digits/./()/+-*/ ) but
        # produce AST nodes outside the numeric-arithmetic set, so the AST
        # walk must still reject them.
        for expr in ("2**10", "1//2"):
            with self.subTest(expr=expr):
                self.assertEqual(self.sum_hours(expr, "test"), -1.0)

    def test_overlong_input_rejected(self):
        from prtime import _HOURS_MAX_LEN
        too_long = "+".join(["1"] * (_HOURS_MAX_LEN))  # well over the cap
        self.assertGreater(len(too_long), _HOURS_MAX_LEN)
        self.assertEqual(self.sum_hours(too_long, "test"), -1.0)


if __name__ == '__main__':
    # unittest.main()
    suite = unittest.TestLoader().loadTestsFromTestCase(Testbasic)
    unittest.TextTestRunner(verbosity=2).run(suite)
