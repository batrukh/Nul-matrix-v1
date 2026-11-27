import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
z=[2,6,14,16,18,20,22,24]; hm=[0.38,0.34,0.23,0.19,0.15,0.11,0.07,0.02]
fig,ax=plt.subplots(figsize=(7.5,4))
ax.plot(z,hm,'o-',c='C0',label='сталі на нитку (природне масштабування)')
ax.axhline(0.37,c='C1',ls='--',label='чинна модель: сталі нормовано на вузол (сліпа до z)')
ax.plot([26,30],[0,0],'x',c='C3',ms=9,label='вікна нема')
for zz,lab in [(2,'1D'),(6,'2D'),(14,'3D'),(30,'4D')]: ax.annotate(f'Кун {lab}\nz={zz}',(zz,0.005),xytext=(zz,0.27 if zz==30 else 0.02),fontsize=8,ha='center')
ax.axvline(25,c='gray',ls=':'); ax.set_xlabel('кількість сусідів вузла z'); ax.set_ylabel('верхня межа вікна h (світ можливий, Матриця стійка)')
ax.set_title('Вікно «жива Матриця + світ» звужується з z і зникає при z ≈ 25',fontsize=10); ax.legend(fontsize=8,loc='upper right'); ax.set_ylim(-0.02,0.45)
plt.tight_layout(); plt.savefig('/mnt/project-files/figures/fig314_window_vs_z.png',dpi=130)
