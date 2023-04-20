file = open("../test/test.py")
lines = []
while True:
    line = file.readlines()
    if not line or line == []:
        break
    lines = lines + line

#123
