import ast,glob,re,numpy as np,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from math import sqrt
R='/mnt/project-files/model/s52/r375/'
def w(k,n,z=1.96):
  p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return c-h,c+h
data={}
for d in glob.glob(R+'a*_T300')+glob.glob('/mnt/project-files/model/s52/r374/a0.*_seed*'):
  a=float(re.search(r'/a([\d.]+)_',d).group(1))
  L=[ast.literal_eval(l) for l in open(d+'/log') if l.startswith('{')]
  L0=L
  if '/r374/' in d: L0=[ast.literal_eval(l) for l in open(d.replace('/r374/','/r373/')+'/log') if l.startswith('{')]+L
  tb=next((x['t'] for x in L0 if x.get('big',0)>=50),None)
  data.setdefault(a,[]).append((L[-1]['big']>=50,tb,L[-1]['big']/4096))
data[0.0]=[(False,None,0)]*8
fig,ax=plt.subplots(1,3,figsize=(13,4))
A=sorted(data)
for a in A:
  v=data[a];k=sum(x[0] for x in v);n=len(v);lo,hi=w(k,n)
  ax[0].errorbar(a,k/n,yerr=[[k/n-lo],[hi-k/n]],fmt='o',color='C0',capsize=4)
  ax[0].text(a,k/n+0.06,f'{k}/{n}',ha='center',fontsize=9)
  for ok,tb,f in v:
    if ok: ax[1].plot(a+np.random.uniform(-.006,.006),tb,'o',color='C1',alpha=.7); ax[2].plot(a+np.random.uniform(-.006,.006),f*100,'o',color='C2',alpha=.7)
ax[0].set(xlabel='хіральність α',ylabel='частка зерен зі світом на t300',title='Вікно народження (95 % інтервал Вілсона)',ylim=(-0.05,1.15))
ax[1].set(xlabel='хіральність α',ylabel='час народження (світ ≥50 вузлів)',title='Коли народжується світ')
ax[2].set(xlabel='хіральність α',ylabel='частка ґратки у світі, %',title='Розмір світу на t300')
fig.suptitle('(375) Кун L16, rq351f, T300; α0 — з (373), лише T100; α0,1 і α0,2 — 16 зерен ((374)+(375)), інші — 8',fontsize=10)
plt.tight_layout();plt.savefig('/mnt/project-files/figures/fig280_3d_vikno_alfa.png',dpi=130)
