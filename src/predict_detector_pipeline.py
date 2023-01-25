
def gen():
    for el in [1,2,3,4,5]:
        yield el

gg = gen()
for g in gg:
    print(g)