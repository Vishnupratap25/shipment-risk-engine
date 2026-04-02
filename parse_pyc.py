import dis
import marshal

f = open(r"c:\Users\9785329\Desktop\new Model\__pycache__\app.cpython-314.pyc", "rb")
f.read(16)
c = marshal.load(f)

out = []
def get_str(co):
    s = [x for x in co.co_consts if isinstance(x, str) and not any(ord(c)>127 for c in x)]
    out.extend(s)
    for x in co.co_consts:
        if hasattr(x, "co_consts"):
            get_str(x)

get_str(c)
with open("strings.txt", "w", encoding="utf-8") as outf:
    outf.write("\n".join(out))
