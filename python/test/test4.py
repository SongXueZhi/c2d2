def testDone(p):
    for prob in p:
        print(prob)
        print(abs(prob-1.0))
        print(min(prob,1))
        if abs(prob-1.0)>1e-6 and min(prob,1)<1.0:
            print(False)
            return False
    print(True)
    return True

testDone([1,0])