import sys,json,ast,numpy as np; sys.path.insert(0,'/mnt/project-files/model/s52')
import fabgen as G
from scipy.sparse import coo_matrix; from scipy.sparse.csgraph import connected_components
L=24; f=G.FabricG(L,"tet"); c=L/2-0.5; d=(f.ijk-c+L/2)%L-L/2; r=np.linalg.norm(d,axis=1)
out={}
for kind in ('C','F'):
  for s in range(1,9):
    D=f'runs/{kind}_L24_seed{s}'; import os; fn=f'{D}/t3_s3_L24_s{s}_ell4.npz'; psi=np.load(fn if os.path.exists(fn) else fn.replace('.npz','_ck.npz'),allow_pickle=True)['psi']
    m=psi>0; e=m[f.tail]&m[f.head]; A=coo_matrix((np.ones(e.sum()),(f.tail[e],f.head[e])),shape=(f.n,f.n))
    nc,lab=connected_components(A,directed=False); lab=np.where(m,lab,-1); sz=np.bincount(lab[m]) if m.any() else np.array([0])
    big=int(sz.max()) if m.any() else 0; W=(lab==np.argmax(sz)) if big>=50 else np.zeros(f.n,bool)
    recs=[ast.literal_eval(l) for l in open(f'{D}/log') if l.startswith('{')]
    tb=next((x['t'] for x in recs if x.get('big',0)>=50),None)
    # place of first birth: ψ>0 centroid not available per time -> use final world centre distance
    o=dict(world=bool(big>=50),size=big,frac_in10=float((r[W]<=10).mean()) if big>=50 else None,frac_in8=float((r[W]<=8).mean()) if big>=50 else None,
           rmax=float(r[W].max()) if big>=50 else None,t_birth=tb,t_end=recs[-1]['t'],nodes_r10=int((r<=10).sum()),nodes_r8=int((r<=8).sum()))
    out[f'{kind}{s}']=o; print(kind,s,o,flush=True)
json.dump(out,open('ana/ana420b.json','w'),indent=1)
