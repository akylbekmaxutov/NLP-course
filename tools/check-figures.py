# -*- coding: utf-8 -*-
"""Detect overlapping / overflowing text in every inline SVG on the site."""
import io, re, glob, sys, html as _html

def texts(svg):
    out=[]
    for m in re.finditer(r'<text\s([^>]*)>(.*?)</text>', svg, re.S):
        at=dict(re.findall(r'(\S+)="([^"]*)"', m.group(1)))
        body=_html.unescape(re.sub(r'<[^>]+>','',m.group(2))).strip()
        if not body: continue
        out.append((float(at.get('x',0)), float(at.get('y',0)),
                    at.get('text-anchor','start'), float(at.get('font-size',12)), body))
    return out

def box(t):
    x,y,anchor,size,body=t
    w=len(body)*size*0.55
    if anchor=='middle': x-=w/2
    elif anchor=='end': x-=w
    return (x, y-size*0.8, x+w, y+size*0.25, body)

def overlap(a,b):
    return not (a[2]<=b[0]+0.5 or b[2]<=a[0]+0.5 or a[3]<=b[1]+0.5 or b[3]<=a[1]+0.5)

problems=0
for f in sorted(glob.glob("language/*/*.html")):
    src=io.open(f,encoding='utf-8').read()
    for m in re.finditer(r'<svg.*?</svg>',src,re.S):
        svg=m.group(0)
        tid=re.search(r'aria-labelledby="([^"]+)"',svg)
        tid=tid.group(1) if tid else "?"
        vb=re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"',svg)
        W,H=(float(vb.group(1)),float(vb.group(2))) if vb else (340,200)
        bs=[box(t) for t in texts(svg)]
        probs=[]
        for i in range(len(bs)):
            for j in range(i+1,len(bs)):
                if overlap(bs[i],bs[j]): probs.append(f"OVERLAP {bs[i][4][:20]!r} x {bs[j][4][:20]!r}")
            if bs[i][2] > W+1: probs.append(f"RIGHT {bs[i][4][:26]!r} ends {bs[i][2]:.0f}>{W:.0f}")
            if bs[i][0] < -1: probs.append(f"LEFT {bs[i][4][:26]!r} starts {bs[i][0]:.0f}")
            if bs[i][3] > H+1: probs.append(f"BOTTOM {bs[i][4][:26]!r}")
        if probs:
            problems+=len(probs)
            print(f"### {f}  {tid}")
            for p in sorted(set(probs))[:6]: print("   ",p)
print(f"\n{problems} layout problems")
sys.exit(1 if problems else 0)
