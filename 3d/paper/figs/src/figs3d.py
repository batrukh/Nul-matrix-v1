"""English figures for the 3D paper (Secs. 4.1-4.6), from existing data only. Run: python3 figs3d.py"""
import ast,glob,re,json,os,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from math import sqrt
plt.rcParams.update({'font.size':8,'axes.titlesize':8,'axes.labelsize':8,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'savefig.dpi':300})
M='/mnt/project-files/model/s52/'; OUT='/mnt/project-files/paper/3d/figs/'
def lab(ax,s): ax.text(-0.16,1.04,s,transform=ax.transAxes,fontweight='bold',fontsize=9)

# ---- Fig. birth window (r373-r375) ----
def wil(k,n,z=1.96):
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return c-h,c+h
def fig_birth():
    data={}
    for d in glob.glob(M+'r375/a*_T300')+glob.glob(M+'r374/a0.*_seed*'):
        a=float(re.search(r'/a([\d.]+)_',d).group(1))
        L=[ast.literal_eval(l) for l in open(d+'/log') if l.startswith('{')]; L0=L
        if '/r374/' in d: L0=[ast.literal_eval(l) for l in open(d.replace('/r374/','/r373/')+'/log') if l.startswith('{')]+L
        tb=next((x['t'] for x in L0 if x.get('big',0)>=50),None)
        data.setdefault(a,[]).append((L[-1]['big']>=50,tb,L[-1]['big']/4096))
    data[0.0]=[(False,None,0)]*8
    rng=np.random.default_rng(1)
    fig,ax=plt.subplots(1,3,figsize=(7,2.3))
    for a in sorted(data):
        v=data[a];k=sum(x[0] for x in v);n=len(v);lo,hi=wil(k,n)
        ax[0].errorbar(a,k/n,yerr=[[k/n-lo],[hi-k/n]],fmt='o',color='C0',capsize=3,ms=4)
        ax[0].text(a+0.012,k/n,f'{k}/{n}',ha='left',va='center',fontsize=6.5)
        for ok,tb,f in v:
            if ok:
                j=rng.uniform(-.006,.006); ax[1].plot(a+j,tb,'o',color='C1',alpha=.7,ms=3); ax[2].plot(a+j,f*100,'o',color='C2',alpha=.7,ms=3)
    ax[0].set(xlabel=r'chiral rotation $\alpha$',ylabel='fraction of seeds with a world',ylim=(-0.05,1.17))
    ax[1].set(xlabel=r'chiral rotation $\alpha$',ylabel=r'birth time (world $\geq50$ nodes)')
    ax[2].set(xlabel=r'chiral rotation $\alpha$',ylabel=r'world at $t=300$, % of nodes',ylim=(0,100))
    for i,a in enumerate(ax): lab(a,'(%s)'%'abc'[i]); a.set_xticks([0,0.05,0.1,0.15,0.2,0.25]); a.set_xlim(-0.02,0.29)
    fig.tight_layout(); fig.savefig(OUT+'fig_birth.png'); plt.close(fig)

# ---- Fig. threshold (r453) ----
def fig_threshold():
    os.chdir(M+'r453')
    fig,(a1,a2)=plt.subplots(1,2,figsize=(7,2.6),gridspec_kw=dict(width_ratios=[1.6,1]))
    for h,col,sst in ((0.36,'#2a6fb0',2.864),(0.30,'#c0392b',2.755)):
        for L,mk in ((16,'o--'),(32,'s-')):
            P=sorted((json.load(open(f))['s0'],json.load(open(f))['modes'][0]['re']) for f in glob.glob(f'eig_s*_h{h:g}_L{L}.json'))
            if P: a1.plot(*zip(*P),mk,color=col,ms=3.5,label=f'$h={h:.2f}$, $L={L}$')
        a1.axvline(sst,color=col,lw=0.9,ls=':'); a1.text(sst+0.005,0.13 if h==0.36 else 0.10,r'$s_0^*=%s$'%sst+'\n(derived)',fontsize=6.5,color=col)
    a1.axhline(0,color='k',lw=0.6); a1.axvspan(2.85,2.95,color='#f2e3c6',alpha=0.6,label='bistable runs')
    a1.set_xlabel(r'budget $s_0$'); a1.set_ylabel(r'$\lambda_{\max}$ ($\rho$–$\psi$ sector)'); a1.legend(loc='lower left'); lab(a1,'(a)')
    N=[];LAM=[]
    for n,f in ((25,'eig_s2.5_h0.36_L32_n25.json'),(100,'eig_s2.5_h0.36_L32.json'),(400,'eig_s2.5_h0.36_L32_n400.json')):
        d=json.load(open(f)); N.append(n); LAM.append(32/np.linalg.norm(d['modes'][0]['kdom']))
    a2.set_xscale('log'); a2.plot(N,LAM,'ko-',ms=4); a2.minorticks_off(); a2.set_xticks(N); a2.set_xticklabels([str(x) for x in N]); a2.set_ylim(8,21)
    a2.set_xlabel(r'range of the conservation kernel $n$'); a2.set_ylabel('wavelength of fastest mode (links)'); lab(a2,'(b)')
    fig.tight_layout(); fig.savefig(OUT+'fig_threshold.png'); plt.close(fig)

# ---- Fig. loop locking (r452 L16, M16) ----
def fig_loops():
    d=json.load(open(M+'r452/L16/ana452v.json'))['rows']; A=sorted(d,key=float)
    x=[float(a) for a in A]
    fig,ax=plt.subplots(1,3,figsize=(7,2.4),sharey=True)
    ax[0].errorbar(x,[d[a]['f2'] for a in A],[d[a]['se'] for a in A],fmt='o-',ms=3.5,label='$L=16$, 8 seeds')
    x8=[float(a) for a in A if d[a]['f2L8'] is not None]; ax[0].plot(x8,[d[a]['f2L8'] for a in A if d[a]['f2L8'] is not None],'s--',color='gray',ms=3.5,label='$L=8$, 2 seeds')
    ax[0].axhline(2/3,color='r',ls=':',lw=1,label='mean-field step')
    os.chdir(M+'r452/M16'); AA=[0,0.05,0.1,0.15,0.2,0.25,0.3,0.4]
    def m(a,c,t0,minn=200):
        v=[json.load(open(f)) for f in glob.glob(f'lock_a{a:g}_s*_L16.json')]
        z=[q[f'lock_{c}2_{t0}'] for q in v if q[f'n_{c}2_{t0}']>=minn and q[f'lock_{c}2_{t0}'] is not None]
        return (np.mean(z),np.std(z,ddof=1)/np.sqrt(len(z))) if len(z)>1 else (np.nan,np.nan)
    y=[m(a,'M',10) for a in AA]; ax[1].errorbar(AA,[q[0] for q in y],[q[1] for q in y],fmt='o-',ms=3.5,color='C0',label='Matrix')
    for c,lb,col in (('M','Matrix','C0'),('F','front','C1'),('W','world','C2')):
        y=[m(a,c,20) for a in AA]; ax[2].errorbar(AA,[q[0] for q in y],[q[1] for q in y],fmt='o-',ms=3.5,color=col,label=lb)
    ax[0].set_ylabel(r'locked fraction of $\sigma=2$ loops'); ax[0].set_ylim(0.2,1.05)
    ax[0].set_title('all loops, $t=10$--15 and 20–25'); ax[1].set_title('before a world, $t=10$--15'); ax[2].set_title('world present, $t=20$--25')
    for i,a in enumerate(ax): a.set_xlabel(r'$\alpha$'); a.axvspan(0.05,0.2,color='g',alpha=0.08); lab(a,'(%s)'%'abc'[i]); a.legend(loc='lower left')
    fig.tight_layout(); fig.savefig(OUT+'fig_loops.png'); plt.close(fig)

# ---- Fig. islands + place set by tau (r425, r449) ----
def fig_islands():
    R=M+'r425/runs'; N=4096
    fig,ax=plt.subplots(1,2,figsize=(7,2.6),gridspec_kw=dict(width_ratios=[1.25,1]))
    cnt={'island':0,'whole box':0,'nothing':0,'growing':0}
    for s in range(1,25):
        D=[ast.literal_eval(l) for l in open(f'{R}/M_seed{s}/log') if l.startswith('{')]
        t=np.array([q['t'] for q in D]); b=np.array([q['big'] for q in D])
        if b[-1]<50: k='nothing'
        elif b[-1]>0.4*N: k='whole box'
        else: k='island' if abs(b[-1]/b[np.argmin(abs(t-300))]-1)<=0.2 else 'growing'
        cnt[k]+=1
        st={'island':dict(color='C2',lw=1.4),'growing':dict(color='C1',lw=1.4),'whole box':dict(color='C0',lw=0.7,alpha=0.6),'nothing':dict(color='0.6',lw=0.7)}[k]
        ax[0].plot(t,100*b/N,**st)
    for k,c in (('island','C2'),('growing','C1'),('whole box','C0'),('nothing','0.6')): ax[0].plot([],[],color=c,label=f'{k} ({cnt[k]}/24)')
    ax[0].axhspan(50/N*100,40,color='C2',alpha=0.07); ax[0].axvspan(300,600,color='k',alpha=0.04)
    ax[0].set_xlabel('time $t$'); ax[0].set_ylabel('largest world, % of nodes'); ax[0].set_ylim(-2,100); ax[0].legend(loc='upper right'); lab(ax[0],'(a)')
    r=json.load(open(M+'r449/ana449b.json')); S=[5,6,7,12]
    for i,s in enumerate(S):
        q=[z for z in r if z['seed']==s]; xs=i+np.linspace(-0.27,0.27,len(q))
        ax[1].bar(xs,[z['null95'] for z in q],width=0.06,color='0.75',label='random shift, 95th pct.' if i==0 else None)
        ax[1].plot(xs,[z['overlap'] for z in q],'o',ms=3.5,color='C2',label='island with new noise' if i==0 else None)
    ax[1].axhline(0.5,ls='--',c='k',lw=0.8)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels([f'arrows {i+1}' for i in range(4)]); ax[1].set_ylim(0,1.3)
    ax[1].set_ylabel('overlap with the original island'); ax[1].legend(loc='upper center',ncol=2,fontsize=6.5); lab(ax[1],'(b)')
    fig.tight_layout(); fig.savefig(OUT+'fig_islands.png'); plt.close(fig); return cnt

# ---- Fig. worlds on grown lattices (r450) ----
def fig_lattices():
    r=json.load(open(M+'r450/ana450.json')); k=np.array([q['seed'] for q in r])
    fig,(a,b)=plt.subplots(1,2,figsize=(7,2.3))
    a.bar(k-0.2,[q['world_on_bcc'] for q in r],0.4,label='bcc nodes')
    a.bar(k+0.2,[np.nan if q['world_on_defects'] is None or q['defect_lat']==0 else q['world_on_defects'] for q in r],0.4,label='defect nodes')
    a.set_xlabel('grown lattice'); a.set_ylabel('fraction of nodes in the world'); a.set_ylim(0,1.15); a.set_xticks(k); a.legend(ncol=2,loc='upper center'); lab(a,'(a)')
    b.bar(k,[100*q['defect_lat'] for q in r],color='0.5'); b.set_xticks(k); b.set_xlabel('grown lattice'); b.set_ylabel('defect nodes (non-bcc), %'); lab(b,'(b)')
    fig.tight_layout(); fig.savefig(OUT+'fig_lattices.png'); plt.close(fig)

if __name__=='__main__':
    fig_birth(); fig_threshold(); fig_loops(); print(fig_islands()); fig_lattices(); print('ok')
