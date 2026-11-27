import ast,numpy as np
R='/mnt/project-files/model/s52/r425/runs'; N=4096
for s in range(1,25):
    D=[ast.literal_eval(l) for l in open(f'{R}/M_seed{s}/log') if l.startswith('{')]
    t=np.array([x['t'] for x in D]); b=np.array([x['big'] for x in D]); A=D[-1]['A2']
    b300=b[np.argmin(abs(t-300))] if t[-1]>=300 else None
    print(s, t[-1], b[-1], b300, round(A,3))
