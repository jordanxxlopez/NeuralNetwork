#!/usr/bin/env python3
"""Check that every original part of a deck is byte-identical in the output and
that original slides keep their relative order.

Usage: python verify_integrity.py ORIGINAL.pptx OUTPUT.pptx
"""
import re
import sys
import zipfile

CHANGED = {"ppt/presentation.xml", "ppt/_rels/presentation.xml.rels", "[Content_Types].xml"}


def slide_order(z):
    rels = z.read("ppt/_rels/presentation.xml.rels").decode()
    rid2t = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="([^"]+)"', rels))
    rid2t.update({a: b for b, a in re.findall(r'Target="([^"]+)"[^>]*Id="(rId\d+)"', rels)})
    pres = z.read("ppt/presentation.xml").decode()
    lst = re.search(r"<p:sldIdLst>(.*?)</p:sldIdLst>", pres, re.S).group(1)
    return [rid2t[r] for r in re.findall(r'r:id="(rId\d+)"', lst)]


def main(orig, out):
    zo, zn = zipfile.ZipFile(orig), zipfile.ZipFile(out)
    bad = [n for n in zo.namelist() if n not in CHANGED
           and (n not in zn.namelist() or zo.read(n) != zn.read(n))]
    o, n = slide_order(zo), slide_order(zn)
    kept = [s for s in n if s in o]
    print(f"original slides: {len(o)}  output slides: {len(n)}  new: {len(n) - len(o)}")
    print("parts identical:", "PASS" if not bad else f"FAIL {bad[:10]}")
    print("original order kept:", "PASS" if kept == o else "FAIL")
    return 0 if not bad and kept == o else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
