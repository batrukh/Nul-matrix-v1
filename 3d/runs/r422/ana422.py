import numpy as np,glob,json
res={}
for a,(t0,t1) in (('0',(2,8)),('0.1',(1,3))):
    S=[];Z=[]
    for f in sorted(glob.glob(f'spec422_L16_s3_seed*_ac{a}.npz')):
        d=np.load(f); t=d['t']; m=(t>=t0-1e-9)&(t<=t1+1e-9); kc=d['kc']
        sig=[0.5*np.polyfit(t[m],np.log(d['sp'][m,j]+1e-300),1)[0] for j in range(1,12)]
        sgz=[0.5*np.polyfit(t[m],np.log(d['sz'][m,j]+1e-300),1)[0] for j in range(1,12)]
        S.append(sig);Z.append(sgz)
    S=np.array(S);Z=np.array(Z);k=kc[1:12]
    am=[int(np.argmax(s)) for s in S]
    o=dict(k=k.tolist(),sig_mean=S.mean(0).tolist(),sig_se=(S.std(0,ddof=1)/np.sqrt(len(S))).tolist(),argmax=am,
           sigz_mean=Z.mean(0).tolist(),argmax_z=[int(np.argmax(s)) for s in Z],n=len(S))
    if a=='0':
        D=-np.polyfit(k**2,S.mean(0),1)[0]; o['D_fit']=float(D); c=np.corrcoef(k**2,S.mean(0))[0,1]; o['r_k2']=float(c)
        low=k<1.5; o['D_low']=float(-np.polyfit(k[low]**2,S.mean(0)[low],1)[0])
    res[a]=o
    print('alpha',a,'n',len(S),'argmax psi',am,'argmax z',o['argmax_z']); print(' k  ',np.round(k,2)); print(' sig',np.round(S.mean(0),3)); print(' se ',np.round(o['sig_se'],3)); print(' sigz',np.round(Z.mean(0),3))
    if a=='0': print(' D',round(o['D_fit'],3),'r',round(o['r_k2'],3),'D_low',round(o['D_low'],3))
json.dump(res,open('ana422.json','w'),indent=1)
