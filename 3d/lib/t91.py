import numpy as np, gauge91 as G
from scipy.linalg import expm
rng=np.random.default_rng(0); L=5
P=G.sym0(rng.standard_normal((L,L,5,5))*0.5); Pi=G.sym0(rng.standard_normal((L,L,5,5))*0.3)
U=np.array([[[expm(0.4*G.PA(rng.standard_normal((5,5)))) for y in range(L)] for x in range(L)] for m in range(3)])
E=np.array([G.PA(rng.standard_normal((L,L,5,5)))*0.3 for m in range(3)])
chi=rng.random((L,L)); wl,wt=G.weights(chi,0.1); a=rng.standard_normal((L,L))
g=G.Gauge(a,wl,wt,ka=1.3,kb=0.7)
Htot=lambda P,U: g.edens(P,0*Pi,U,0*E).sum()
FP,FE=g.forces(P,U)
h=1e-6; d=G.sym0(rng.standard_normal((L,L,5,5)))
num=(Htot(P+h*d,U)-Htot(P-h*d,U))/(2*h); print('Phi',num, -(FP*d).sum())
for m in range(3):
  x,y=2,3; Gm=G.gen(1,4)+0.3*G.gen(0,2)
  Up=U.copy(); Up[m,x,y]=expm(h*Gm)@U[m,x,y]; Um=U.copy(); Um[m,x,y]=expm(-h*Gm)@U[m,x,y]
  num=(Htot(P,Up)-Htot(P,Um))/(2*h); R=-FE[m,x,y]*(g.ka*wl[m,x,y])/2
  print('U',m,num,(R*Gm).sum())
# енергія
H0=g.edens(P,Pi,U,E).sum()
for k in range(400): P,Pi,U,E=g.step(P,Pi,U,E,0.01)
print('energy',H0,g.edens(P,Pi,U,E).sum())
