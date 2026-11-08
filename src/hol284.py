import os as _o; _V = _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), 'variant'); RUNS = _o.environ.get('NM_RUNS', _o.path.join(_o.path.dirname(_o.path.abspath(__file__)), '..', 'runs', 'out', 'variant'))
import sys,glob,pickle,json,numpy as np
sys.path.insert(0,_V)
import phi90 as PH, matter98 as M98
L=96
def plane(P):
    w,v=np.linalg.eigh(P); return v[:,3:5]
def loop_sign(P,cx,cy,R):
    X,Y=PH.coords(L)
    pts=[]
    for a in np.linspace(0,2*np.pi,int(12*R),endpoint=False):
        r,_,_=M98.wrapd(X,Y,cx+R*np.cos(a),cy+R*np.sin(a),L); pts.append(np.unravel_index(np.argmin(r),r.shape))
    pts=[p for i,p in enumerate(pts) if i==0 or p!=pts[i-1]]
    if pts[-1]==pts[0]: pts=pts[:-1]
    F0=plane(P[pts[0]]); F=F0.copy(); minov=1
    for p in pts[1:]+[pts[0]]:
        Fn=plane(P[p]); O=F.T@Fn            # перенос рамки: найближчий поворот
        u,s,vt=np.linalg.svd(O); minov=min(minov,s.min()); F=Fn@(vt.T@u.T)
    return float(np.linalg.det(F0.T@F)), minov
for seed in [1,2,3,4]:
    fs=glob.glob(RUNS + f'/g0.0025_s{seed}_T600/*_fine.json')
    cks=glob.glob(RUNS + f'/ck/g0.0025_s{seed}_T600/*_ck.pkl')
    if not fs or not cks: print(seed,'нема'); continue
    F=json.load(open(fs[0])); C=pickle.load(open(cks[0],'rb')); P=C['GF'].P; psi=C['psi'].reshape(L,L)
    tck=C.get('t',None)
    old=[c for t_,cl in F if 585<=t_<=590 for c in cl]
    print('seed',seed,'t_ck',tck,'вихорів',len(old))
    X,Y=PH.coords(L)
    for (ox,oy) in old[:6]:
        print('  вихор',(round(ox,1),round(oy,1)),[ (R,)+tuple(round(x,3) for x in loop_sign(P,ox,oy,R)) for R in (2,3,4)])
    # контроль: петлі в глибині світу без вихорів
    dist=None
    import scipy.ndimage as ndi
    dist=ndi.distance_transform_edt(np.tile(psi>0,(3,3)))[L:2*L,L:2*L]
    ok=dist>=5
    for (ox,oy) in old:
        r,_,_=M98.wrapd(X,Y,ox,oy,L); ok&=r>=8
    idx=np.argwhere(ok)[::max(1,len(np.argwhere(ok))//3)][:3]
    for a,b in idx: print('  контроль',(round(X[a,b],1),round(Y[a,b],1)),loop_sign(P,X[a,b],Y[a,b],3))
