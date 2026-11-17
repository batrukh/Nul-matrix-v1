import numpy as np, numba as nb
@nb.njit(cache=True, fastmath=True)
def qforce2(z, E, S, QV, chi, dchi, eps0, d2, a, m, n):
    k, nq = E.shape; lw = np.zeros(m, np.complex128); br = np.zeros(m, np.complex128); fc = np.zeros(n)
    for q in range(nq):
        e0=E[0,q]; e1=E[1,q]; e2=E[2,q]; e3=E[3,q]
        l0=z[e0]; l1=z[e1]; l2=z[e2]; l3=z[e3]
        if S[0,q]<0: l0=l0.conjugate()
        if S[1,q]<0: l1=l1.conjugate()
        if S[2,q]<0: l2=l2.conjugate()
        if S[3,q]<0: l3=l3.conjugate()
        n0=QV[q,0]; n1=QV[q,1]; n2=QV[q,2]; n3=QV[q,3]
        p01=l0*l1; p23=l2*l3; P=p01*p23
        Et=eps0*(chi[n0]+chi[n1]+chi[n2]+chi[n3])*0.25
        aP2=P.real*P.real+P.imag*P.imag+d2; aP=np.sqrt(aP2)
        pa=1.0/np.sqrt(aP) if a==0.75 else aP**(a-1)   # a=0.75: aP**(-0.25)
        if a==0.75: pa=1.0/np.sqrt(np.sqrt(aP))
        rr=P.real*P/aP2
        cB=pa*(1+(a-1)*rr); cE=-Et*pa*(a*P/aP-1-(a-1)*rr)
        cost=(aP*pa-P.real*pa)*0.25*eps0
        Q0=(l1*p23).conjugate(); Q1=(l0*p23).conjugate(); Q2=(p01*l3).conjugate(); Q3=(p01*l2).conjugate()
        c=Q0*cB; d=Q0*cE
        if S[0,q]<0: c=c.conjugate(); d=d.conjugate()
        lw[e0]+=c; br[e0]+=d
        c=Q1*cB; d=Q1*cE
        if S[1,q]<0: c=c.conjugate(); d=d.conjugate()
        lw[e1]+=c; br[e1]+=d
        c=Q2*cB; d=Q2*cE
        if S[2,q]<0: c=c.conjugate(); d=d.conjugate()
        lw[e2]+=c; br[e2]+=d
        c=Q3*cB; d=Q3*cE
        if S[3,q]<0: c=c.conjugate(); d=d.conjugate()
        lw[e3]+=c; br[e3]+=d
        fc[n0]+=dchi[n0]*cost; fc[n1]+=dchi[n1]*cost; fc[n2]+=dchi[n2]*cost; fc[n3]+=dchi[n3]*cost
    return lw, br, fc
