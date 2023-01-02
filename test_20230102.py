# %%
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

# %%
wna = np.arange(0., 90., 1.)
phi = 0

# %%
# 栗田さんIDLコード転記
wna_rad=wna / 180 * np.pi #[degree] to [rad]
phi_rad=phi / 180 * np.pi #[degree] to [rad]

vc=3.0e8 #speed of light

wce=2800*2*np.pi #!dpi=Pi
wpe=5.0*wce #assume wpe/wce=5.0
wpi=wpe/np.sqrt(1840.0)
wci=wce/1840.0
fsamp=8192.0*8 # 65kHz OFA

ww=0.2*wce # + dfreq[i]  #assume 0.2wce wave

# Cold plasma dispersion relation is evaluated
# assuming plasma consisting of electrons and protons

sp=1.0-wpe*wpe/(ww*ww-wce*wce)-wpi*wpi/(ww*ww-wci*wci) # Stix S parameter
dp=-wce*wpe*wpe/(ww**3-ww*wce*wce)+wpi*wpi*wci/(ww**3-ww*wci*wci) # Stix D parameter
pp=1.0-wpe*wpe/ww**2-wpi*wpi/ww**2 # Stix P parameter

rr=sp+dp
ll=sp-dp

wna_=wna_rad
phi_=phi_rad

aa=sp*np.sin(wna_)*np.sin(wna_)+pp*np.cos(wna_)*np.cos(wna_)
bb=rr*ll*np.sin(wna_)*np.sin(wna_)+pp*sp*(1.0+np.cos(wna_)*np.cos(wna_))
ff=np.sqrt(((rr*ll-pp*sp)**2)*np.sin(wna_)**4+4*(pp*dp*np.cos(wna_))**2)

nr=np.sqrt((bb-ff)/(2.0*aa))

# For test use
# Square root of approximate dispersion relation for whistler-mode waves
nr_test=np.sqrt(wpe*wpe/(ww*(wce*np.cos(wna_)-ww))) # ぶんぼ近似バージョン
# nr_test = np.sqrt(1 - wpe*wpe/(ww*(ww-wce*np.cos(wna_)))) # 分母近似なしバージョン

# %%
# 栗田さんコードからdp修正
# Cold plasma dispersion relation is evaluated
# assuming plasma consisting of electrons and protons

sp=1.0-wpe*wpe/(ww*ww-wce*wce)-wpi*wpi/(ww*ww-wci*wci) # Stix S parameter
# dp=-wce*wpe*wpe/(ww**3-ww*wce*wce)+wpi*wpi*wci/(ww**3-ww*wci*wci) # Stix D parameter
dp_new=wce*wpe*wpe/(ww**3-ww*wce*wce)+wpi*wpi*wci/(ww**3-ww*wci*wci) # Stix D parameter
pp=1.0-wpe*wpe/ww**2-wpi*wpi/ww**2 # Stix P parameter

rr=sp+dp_new
ll=sp-dp_new

wna_=wna_rad
phi_=phi_rad

aa=sp*np.sin(wna_)*np.sin(wna_)+pp*np.cos(wna_)*np.cos(wna_)
bb=rr*ll*np.sin(wna_)*np.sin(wna_)+pp*sp*(1.0+np.cos(wna_)*np.cos(wna_))
ff=np.sqrt(((rr*ll-pp*sp)**2)*np.sin(wna_)**4+4*(pp*dp_new*np.cos(wna_))**2)

nr_new=np.sqrt((bb-ff)/(2.0*aa))


# %%
# print(nr, nr_test, nr_new)
plt.figure(figsize=(8, 6))
plt.subplot(2,1,1)
plt.plot(wna, nr, label='n')
# plt.plot(wna, nr_test, label='n_stix for test')
plt.plot(wna, nr_new, label='n_new')
plt.xlabel('wna [deg]')
plt.ylabel('refractive index')
plt.legend()
plt.subplot(2,1,2)
# plt.plot(wna, (nr-nr_test)/nr_test*100, label='(n - n_stix) / n_stix')
# plt.plot(wna, (nr_new-nr_test)/nr_test*100, label='(n_new - n_stix) / n_stix')
plt.plot(wna, (nr-nr_new)/nr_new*100, label='(n - n_new) / n_new')
plt.xlabel('wna [deg]')
plt.ylabel('deff [%]')
plt.tight_layout()
plt.legend()
plt.show()

# %%
print(dp, dp_new)

# %%

examp=1.0 
eyamp=dp/(sp-nr*nr) 
eyamp_new=dp_new/(sp-nr*nr) 
ezamp=-nr*nr*np.cos(wna_)*np.sin(wna_)/(pp-nr*nr*np.sin(wna_)*np.sin(wna_)) 

bxamp=-nr*np.cos(wna_)*dp/vc/(sp-nr*nr) 
bxamp_new=-nr*np.cos(wna_)*dp_new/vc/(sp-nr*nr) 
byamp=nr*np.cos(wna_)*pp/vc/(pp-nr*nr*np.sin(wna_)*np.sin(wna_)) 
bzamp=nr*np.sin(wna_)*dp/vc/(sp-nr*nr) 
bzamp_new=nr*np.sin(wna_)*dp_new/vc/(sp-nr*nr) 

plt.plot(eyamp)
plt.plot(eyamp_new)
plt.show()

plt.plot(bxamp)
plt.plot(bxamp_new)
plt.show()

plt.plot(bzamp)
plt.plot(bzamp_new)
plt.show()


# %%
