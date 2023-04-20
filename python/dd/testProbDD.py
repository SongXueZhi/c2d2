import debuggingbook.DeltaDebugger as DT

while True:
    fuzz_input = DT.fuzz()
    try:
        DT.mystery(fuzz_input)
    except ValueError:
        break
DT.ddmin(DT.code_reduce_test,fuzz_input,DT.mystery,ValueError('Invalid input'))
DT.ProbDD(DT.code_reduce_test, fuzz_input, DT.mystery, ValueError('Invalid input'))
