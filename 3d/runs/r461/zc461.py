import sys,numpy as np; sys.path.insert(0,"/mnt/project-files/model/s52/r454"); import front454 as F
w2=F.w2; F.g=1.3/(2*w2)
for z in (14,16,18,20,22,24,26,28):
    F.c=1.3*z/(4*w2); hs=[]
    for h in np.arange(0,0.5,0.01):
        try: st=F.state(3.0,h,3.0)
        except Exception: st=None
        if st and st['det']>0 and st['I']>0: hs.append(h)
    print(z, f"{max(hs):.2f}" if hs else "нема", flush=True)
