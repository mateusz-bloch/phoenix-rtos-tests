from typing import Optional

from trunner.ctx import TestContext
from trunner.dut import Dut
from trunner.types import Status, TestResult
import psh.tools.psh as psh

def harness(dut: Dut, ctx: TestContext, result: TestResult, **kwargs) -> Optional[TestResult]:
    current_test = None
    stats = {"PASS": 0, "FAIL": 0}
    timeout = 60 if ctx.nightly else 20
    test = kwargs.get("test", None)
    if None:
        return TestResult(status=Status.FAIL)

    test_start_re   = r"Testing (?P<name>\w+)"
    fail_expr_re    = r".* failed at (?P<file>[^:]+):(?P<line>\d+)"
    fail_msg_re     = r"FAIL: (?P<msg>.*)"
    summary_re      = r"(?P<passed>\d+)/(?P<total>\d+) tests passed"
    banner_pass_re  = r"~~~ALL TESTS PASSED~~~"
    banner_fail_re  = r"~~~SOME TESTS FAILED~~~"

    psh._send(dut, f"{test}")

    while True:
        idx = dut.expect(
            [
                test_start_re,   # 0
                fail_expr_re,    # 1
                fail_msg_re,     # 2
                summary_re,      # 3
                banner_pass_re,  # 4
                banner_fail_re   # 5
            ],
            timeout=timeout,
        )
        gd = dut.match.groupdict()

        if idx == 0:
            current_test = gd["name"]
            result.add_subresult(current_test, Status.OK, "")
            stats["PASS"] += 1

        elif idx in (1, 2):
            if current_test is None:
                continue

            if result.subresults[current_test].status != Status.FAIL:
                stats["PASS"] -= 1
                stats["FAIL"] += 1
                msg = (
                    gd.get("msg")
                    or f"failed at {gd.get('file')}:{gd.get('line')}"
                    or "assertion failed"
                )
                result.add_subresult(current_test, Status.FAIL, msg)

        elif idx == 3:
            passed, total = int(gd["passed"]), int(gd["total"])
            assert total == stats["PASS"] + stats["FAIL"], (
                "Rozbieżność: parser widzi "
                f"{stats['PASS'] + stats['FAIL']} testów, "
                f"a podsumowanie podaje {total}."
            )

            status = Status.OK if stats["FAIL"] == 0 else Status.FAIL
            return TestResult(status=status)

        elif idx == 4:
            return TestResult(status=Status.OK)

        elif idx == 5:
            return TestResult(status=Status.FAIL)
