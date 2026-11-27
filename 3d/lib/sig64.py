import numpy as np, os
os.environ.setdefault("DELTA","1e-6")
import simNlib as N, rnd51 as R, local60 as A
L=96
for (ell,g,sd) in [(8,0,1),(8,0,2),(4,0,3),(16,0,4),(8,0.02,1),(8,0.2,1),(0,1,1)]:
    f=N.Fabric(L); tau=A.tau_field(L,ell,sd) if ell else np.zeros(f.n)
    x_,y_=f.xy[f.tail].T; x2,y2=f.xy[f.head].T
    Gg=g*(((x2-x_+L//2)%L-L//2)+0.31*((y2-y_+L//2)%L-L//2))
    bad=tau[f.head]-tau[f.tail]+Gg<0; R.flip(f,bad)
    S=f.S1+f.S2+f.S3; sig=np.sign(S)
    print(ell,g,sd,"ntri",f.ntri,"cyclic",int((abs(S)==3).sum()),"sum sigma",int(sig.sum()), "half0", int(sig[::2].sum()), int(sig[1::2].sum()))
