import sys,json,numpy as np,glob
sys.path.insert(0,'../r421'); sys.path.insert(0,'.')
import classify as C, fabpts as F
out={}
for f in sorted(glob.glob('runs/lat_[SQ]_s?.npz')):
    d=np.load(f); P=d['P']; box=float(d['box'])
    fab=F.FabricP.fromPoints(P,box) if hasattr(F.FabricP,'fromPoints') else F.fromPoints(P,box)
    deg=np.asarray(fab.deg)
    fr,lab=C.cna(P,box)
    r=dict(n=len(P),deg14=float(np.mean(deg==14)),cna=fr)
    out[f]=r; print(f,r,flush=True)
json.dump(out,open('runs/ana_lat.json','w'),indent=1)
