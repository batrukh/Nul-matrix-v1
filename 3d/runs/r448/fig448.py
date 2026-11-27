import sys,json,numpy as np,matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,'../r421'); import classify as C
plt.rcParams['axes.formatter.use_locale']=False
fmt=matplotlib.ticker.FuncFormatter(lambda x,p:f'{x:g}'.replace('.',','))
R=[json.load(open(f'runs/R_e0.25_s{s}.json'))['final'] for s in range(1,9)]
Z=[json.load(open(f'runs/Z_e0.25_s{s}.json'))['final'] for s in range(1,9)]
fig=plt.figure(figsize=(13,4.6))
a=fig.add_subplot(131); x=np.arange(1,9)
a.bar(x-0.2,[100*r['bcc'] for r in R],0.4,label='повернутий зародок (R)',color='#c0504d')
a.bar(x+0.2,[100*r['bcc'] for r in Z],0.4,label='зародок за осями (Z)',color='#4f81bd')
a.axhline(90,ls='--',c='k',lw=1); a.text(0.6,90.5,'поріг 90 %',fontsize=8)
a.set_ylim(80,101); a.set_xlabel('зерно'); a.set_ylabel('частка ОЦК (CNA), %'); a.legend(fontsize=8,loc='lower right'); a.set_title('Частка ОЦК, t ≤ 2000'); a.yaxis.set_major_formatter(fmt)
b=fig.add_subplot(132)
b.bar(x-0.2,[100*(1-r['deg14']) for r in R],0.4,color='#c0504d'); b.bar(x+0.2,[100*(1-r['deg14']) for r in Z],0.4,color='#4f81bd')
b.axhline(2,ls='--',c='k',lw=1); b.text(0.6,2.1,'поріг 2 %',fontsize=8)
b.axhline(1.66,ls=':',c='g',lw=1); b.text(0.6,1.3,'поріг світу 1,66 % (443Б)',fontsize=8,color='g')
b.set_xlabel('зерно'); b.set_ylabel('вузли з числом сусідів ≠ 14, %'); b.set_title('Дефекти'); b.yaxis.set_major_formatter(fmt)
c=fig.add_subplot(133,projection='3d')
z=np.load('runs/R_e0.25_s1.npz'); P=z['P']; L=float(np.atleast_1d(z['box'])[0]); fr,lab=C.cna(P,L); lab=np.asarray(lab)
D=P[lab!='bcc']%L; c.scatter(*D.T,s=4,c='#c0504d'); c.set_xlim(0,L);c.set_ylim(0,L);c.set_zlim(0,L)
c.set_title('Зерно 1 (R): дефекти — тонкий шар через ящик',fontsize=9); c.set_xticklabels([]);c.set_yticklabels([]);c.set_zticklabels([])
fig.suptitle('(448) Ріст ОЦК із зародка, ε 0,25; без зародка — рідина 8/8',fontsize=11)
plt.tight_layout(); plt.savefig('/mnt/project-files/figures/fig300_3d_pfc_zarodok.png',dpi=130)
