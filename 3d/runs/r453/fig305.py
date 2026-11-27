"""fig305: (453) найбільше власне число сектора «ρ ↔ ψ» від запасу s0 (Кун L16, L32; h 0,36 і 0,30) і виведені пороги s0*; б) довжина хвилі найшвидшої моди від дальності K (n)."""
import json, glob, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,(a1,a2)=plt.subplots(1,2,figsize=(9,3.8),dpi=150,gridspec_kw=dict(width_ratios=[1.6,1]))
for h,col,sst in ((0.36,'#2a6fb0',2.864),(0.30,'#c0392b',2.755)):
    for L,mk in ((16,'o--'),(32,'s-')):
        P=[]
        for f in glob.glob(f'eig_s*_h{h:g}_L{L}.json'):
            d=json.load(open(f)); P.append((d['s0'],d['modes'][0]['re']))
        P.sort(); 
        if P: a1.plot(*zip(*P),mk,color=col,ms=4,label=f'h {h:g}, L{L}')
    a1.axvline(sst,color=col,lw=0.9,ls=':'); a1.text(sst+0.005,0.13 if h==0.36 else 0.10,f's0* = {sst}\n(вивід)',fontsize=7,color=col)
a1.axhline(0,color='k',lw=0.6); a1.axvspan(2.85,2.95,color='#f9eeda',alpha=0.6,label='бістабільність (381)')
a1.set_xlabel('запас s0'); a1.set_ylabel('λ_max (сектор ρ ↔ ψ)'); a1.legend(fontsize=7,loc='lower left'); a1.set_title('а) поріг нестійкості Матриці',fontsize=9)
N=[];LAM=[]
for n,f in ((25,'eig_s2.5_h0.36_L32_n25.json'),(100,'eig_s2.5_h0.36_L32.json'),(400,'eig_s2.5_h0.36_L32_n400.json')):
    d=json.load(open(f)); k=np.linalg.norm(d['modes'][0]['kdom']); N.append(n); LAM.append(32/k)
a2.set_xscale('log'); a2.plot(N,LAM,'ko-',ms=5); a2.minorticks_off(); a2.set_xlabel('дальність закону збереження n (K = Pⁿ)'); a2.set_ylabel('довжина хвилі, нитки'); a2.set_title('б) довжина найшвидшої моди (s0 2,5)',fontsize=9)
a2.set_xticks(N); a2.set_xticklabels([str(x) for x in N]); a2.set_ylim(8,21)
fig.suptitle('(453) Система z–ψ: поріг s0* виведено і підтверджено; власна довжина є лише нижче порогу',fontsize=10); fig.tight_layout()
fig.savefig('/mnt/project-files/figures/fig305_zpsi_threshold.png'); print('ok', N, LAM)
