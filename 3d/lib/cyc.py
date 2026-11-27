import sys, numpy as np
sys.path.insert(0, '.')
import stall43 as T, rnd51 as R, seam51 as SM
L=48
def fab():
    s=T.Sys(L=L); f=s.fab; R.flip(f, SM.seams(f)); return f
def masks(f):
    xa,ya=f.xy[f.tail].T; xb,yb=f.xy[f.head].T
    dy=(yb-ya+L//2)%L-L//2; dx=(xb-xa+L//2)%L-L//2; mx=np.minimum(xa,xb); my=np.minimum(ya,yb)
    H=(dy==0)&(np.abs(dx)==1); V=(dx==0)&(np.abs(dy)==1); return H,V,mx,my,xa,ya
def cyc(f, bad):
    R.flip(f,bad); S=np.stack([f.S1,f.S2,f.S3],1); c=np.where(np.abs(S.sum(1))==3)[0]
    R.flip(f,bad); return [(tuple(f.xy[f.TV[i]].mean(0).round(1)), int(np.sign(S[i].sum()))) for i in c]
f=fab(); H,V,mx,my,xa,ya=masks(f); c=L//2
print('uniform', cyc(f, np.zeros(f.m,bool)))
for l in (12,24):
    print('segment',l, cyc(f, H&(ya==c)&(mx>=c-l//2)&(mx<c-l//2+l)))
print('full row', cyc(f, H&(ya==c)))
a=24; x0=c-a//2; y0=x0
print('loop', cyc(f, H&np.isin(ya,(y0,y0+a))&(mx>=x0)&(mx<x0+a)|V&np.isin(xa,(x0,x0+a))&(my>=y0)&(my<y0+a)))
print('cross full', cyc(f, H&(ya==c)|V&(xa==c)))
print('vseg 24', cyc(f, V&(xa==c)&(my>=c-12)&(my<c+12)))
