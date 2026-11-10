import numpy as np, sys, glob
for f in sorted(glob.glob('disp_ka*.npz')):
    d = np.load(f); modes = d['modes']; L = 48; dt = 0.25
    print(f)
    for comp in ['E34', 'P01', 'P34', 'P03', 'P00']:
        rec = d[comp]; t = np.arange(len(rec))*dt; ks = []; ws = []
        for k, (n, m) in enumerate(modes):
            kx = 2*np.pi*n/L; ky = (2*np.pi*m/L - 0.5*kx)/(np.sqrt(3)/2); kk = np.hypot(kx, ky)
            x = rec[:, k]*np.hanning(len(t)); F = np.abs(np.fft.fft(x, 16*len(x))); w = 2*np.pi*np.fft.fftfreq(16*len(x), dt)
            sel = (np.abs(w) > 0.05) & (np.abs(w) < 3); i = np.argmax(F*sel); ks.append(kk); ws.append(abs(w[i]))
        ks, ws = np.array(ks), np.array(ws); o = np.argsort(ks)
        p = np.polyfit(ks[o][:5]**2, ws[o][:5]**2, 1)
        print(f'  {comp}: ' + ' '.join(f'{k:.2f}:{w:.3f}' for k, w in zip(ks[o], ws[o])) + f' | fit c={np.sqrt(max(p[0],0)):.3f} m={np.sqrt(max(p[1],0)):.3f}')
