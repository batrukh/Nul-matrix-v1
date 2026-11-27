"""(461) вивід: як вікно «жива Матриця + світ можливий» у зведеній теорії (454) залежить від числа сусідів z.
Чинна модель: сталі на вузол нормовано (lc·6/deg, kappa·6/deg, eps0·8/tq) -> c і g у (454) від z не залежать (сліпа до вимірності).
Природне масштабування (сталі на нитку як у 2D): c(z)=1,3·z/(4w²), g=1,3/(2w²).
Вікно: при rM=s0 Матриця стійка (det>0), стан світу існує, і фронт світу іде вперед (I>0) — метастабільна Матриця, світ може народитись і рости."""
import sys, json, numpy as np
sys.path.insert(0,"/mnt/project-files/model/s52/r454"); import front454 as F
w2=F.w2; out={}
for mode in ("norm","natural"):
    for z in (2,6,14,30):
        F.c = 1.3*6/(4*w2) if mode=="norm" else 1.3*z/(4*w2)
        F.g = 1.3*6/14/(2*w2) if mode=="norm" else 1.3/(2*w2)
        win=[]
        for h in np.round(np.arange(0.0,1.21,0.01),2):
            try: st=F.state(3.0,h,3.0)
            except Exception: st=None
            ok = bool(st) and st['det']>0 and st['I']>0
            win.append((float(h),ok, None if not st else round(st['I'],3)))
        hs=[h for h,ok,_ in win if ok]
        out[f"{mode}_{z}"]=dict(c=F.c,g=F.g,hmin=min(hs) if hs else None,hmax=max(hs) if hs else None,n=len(hs))
        print(mode,"z",z,"c %.3f g %.4f"%(F.c,F.g),"вікно h:",(f"{min(hs):.2f}–{max(hs):.2f} ({len(hs)} точок)" if hs else "НЕМА"),flush=True)
json.dump(out,open("dim461.json","w"),indent=1)
