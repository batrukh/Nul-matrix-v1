"""(319) склад легкої моди директора при k=0: у які власні вектори Φ повертається n. python3 comp.py VACTAG MU"""
import sys, numpy as np
sys.argv = ['disp.py', sys.argv[1], sys.argv[2], '0.09']; exec(open('/mnt/project-files/model/s52/r319/disp.py').read().split("Gph = np.zeros")[0])
w2, V = omega(0.0, 0.0); ev, evec = np.linalg.eigh(Pv)
for i in np.argsort(w2)[:10]:
    v = V[:, i]; dn = BT.T @ v[14:18]; c = np.abs(evec.T @ dn)**2
    print(f"ω {np.sqrt(abs(w2[i])):.4f} n-вага {np.sum(abs(v[14:18])**2):.2f} δn у власних векторах Φ {np.round(ev,3)}: {np.round(c/max(c.sum(),1e-30),3)}")
