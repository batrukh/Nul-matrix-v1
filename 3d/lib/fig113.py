import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R=json.load(open("sel113.json")); fig,ax=plt.subplots(1,2,figsize=(13,4.6))
bins=[(3,10),(10,30),(30,60),(60,120),(120,10000)]; lab=["3–9","10–29","30–59","60–119","≥ 120"]
P=[];Nn=[]
for lo,hi in bins:
    q=[r for r in R if lo<=r["size"]<hi]; P.append(100*sum(r["born"] for r in q)/max(len(q),1)); Nn.append(len(q))
ax[0].bar(range(5),P,color=["#7f8c8d"]*3+["#2a6fb0","#c0392b"])
for i,(p,n) in enumerate(zip(P,Nn)): ax[0].text(i,p+0.05,f"{n} кадрів",ha="center",fontsize=8)
ax[0].set_xticks(range(5)); ax[0].set_xticklabels(lab); ax[0].set_xlabel("розмір протосвіту (вузлів)"); ax[0].set_ylabel("стає світом у наступні 5 од., %")
ax[0].set_title("Світ народжується лише з протосвіту ≥ 95 вузлів")
big=[r for r in R if r["size"]>=60]
for born,c,m,l in ((False,"#bbbbbb","o","великий, світом не став"),(True,"#c0392b","*","став світом")):
    q=[r for r in big if r["born"]==born]; ax[1].scatter([r["hol"] for r in q],[r["psi"] for r in q],s=[20 if not born else 160 for _ in q],c=c,marker=m,label=l,edgecolors="k" if born else "none",linewidths=0.5)
ax[1].set_xlabel("закрученість ниток протосвіту"); ax[1].set_ylabel("середнє ψ протосвіту"); ax[1].set_title("Серед великих — ті, де порядок глибший і ψ вище"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig27_protoworld_selection.png",dpi=130)
