import time
from typing import Optional, Type

import debuggingbook.DeltaDebugger as DT


def code_reduce(filename: str, exception: Optional[Type] = None):
    file = open("../test/" + filename + ".py")
    lines = []
    while True:
        line = file.readlines()
        if not line or line == []:
            break
        lines = lines + line

    print("\n------result_ddmin-------")
    tic1 = time.perf_counter_ns()
    result_ddmin = DT.ddmin(DT.code_reduce_test, lines, DT.run, exception)
    toc1 = time.perf_counter_ns()
    print(f"用时: {(toc1 - tic1)}")
    DT.print_content("".join(result_ddmin))

    print("\n------result_probdd-------")
    tic2 = time.perf_counter_ns()
    result_probdd = DT.ProbDD(DT.code_reduce_test, lines, DT.run, exception)
    toc2 = time.perf_counter_ns()
    print(f"用时: {(toc2 - tic2)}")
    DT.print_content("".join(result_probdd))

testdict = {
    "test1": AssertionError("My Test"),
    "test2": AssertionError("My Test"),
    "test3": FileNotFoundError("[Errno 2] No such file or directory: '../test/test.py'"),
    "test4": NameError("name 'd' is not defined")
}

for test in testdict:
    print(f"{test}:")
    code_reduce(test, testdict.get(test))

