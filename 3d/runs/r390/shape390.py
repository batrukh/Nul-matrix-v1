"""(390) форма зупинених світів на правому краї вікна (α0,2): обгортка тора, виснаження Матриці. Код обгортки — з r384/ana384.py."""
import sys, json; sys.argv=['x']; sys.path[:0]=['/mnt/project-files/model/s52/r351','/mnt/project-files/model/s52']
exec(open('/mnt/project-files/model/s52/r384/ana/ana384.py').read().split('out = []')[0])
B='/mnt/project-files/model/s52/'
runs={'r375 α0,2 з12':'r375/a0.2_seed12_T300/t3_s3_L16_s12_ell4.npz','r374 α0,2 з7':'r374/a0.2_seed7/t3_s3_L16_s7_ell4.npz',
      'контроль r375 α0,2 з9':'r375/a0.2_seed9_T300/t3_s3_L16_s9_ell4.npz','контроль r375 α0,15 з1':'r375/a0.15_seed1_T300/t3_s3_L16_s1_ell4.npz'}
res={}
for k,fn in runs.items():
    c=np.load(B+fn); z=c['z']; psi=c['psi']; A=np.abs(z)**2
    nw=psi>0; wl=nw[T]&nw[H]; lab,sz=comps(wl,nw); bl=sz.argmax(); inb=nw&(lab==bl)
    far=(~wl)&~(inb[T]|inb[H]); edge=(~wl)&(inb[T]|inb[H])
    r=dict(big=int(inb.sum()),frac=round(inb.sum()/N,3),wrap=wraps(inb,wl&inb[T]&inb[H]),
           Afar=round(float(A[far].mean()),3) if far.any() else None, Aedge=round(float(A[edge].mean()),3),
           psiM=round(float(psi[~nw].mean()),3))
    res[k]=r; print(k,r)
json.dump(res,open('shape390.json','w'),ensure_ascii=False)
