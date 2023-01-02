# %%
import matplotlib.pyplot as plt
import numpy as np


# %%
Ne = np.arange(0., 1000., 10.)
fp = 8.97 * np.sqrt(Ne)
plt.figure(figsize=(15, 8))
plt.subplot(1, 2, 1)
plt.plot(Ne, fp)
plt.subplot(1, 2, 2)
plt.plot(fp/2, fp)
plt.plot(fp/10, fp)
# fpは少なくて90 kHz、多くて180 kHz程度
# fc = 2-10 kHz程度と思うと
# fp/fcは少なくて90/2=45,90/10=9、多くて180/2=90,180/10=18 程度

# %%


def appleton_hartree(f, q, theta):
    rp2 = f**2 - (2.*q**2*(f**2-q**2)*f) / (2.*f*(f**2-q**2)-f*np.sin(theta)**2+np.sqrt(f**2*np.sin(theta)**4+4*(f**2-q**2)**2*np.cos(theta)**2))
    return np.sqrt(rp2)


def appleton_hartree_(f, q, theta):
    rm2 = f**2 - (2.*q**2*(f**2-q**2)*f) / (2.*f*(f**2-q**2)-f*np.sin(theta)**2-np.sqrt(f**2*np.sin(theta)**4+4*(f**2-q**2)**2*np.cos(theta)**2))
    return np.sqrt(rm2)


def appleton_hartree_quasi_parallel(f, q, theta):
    rp2 = f**2 - (q**2*f) / (f-np.cos(theta))
    return np.sqrt(rp2)


def appleton_hartree_quasi_parallel_(f, q, theta):
    rm2 = f**2 - (q**2*f) / (f+np.cos(theta))
    return np.sqrt(rm2)


# %%
f = np.arange(0., 1.0, 0.000001)  # w/wc
# q = np.array([0.1, 1., 10.])  # wp/wc
q = np.array([1., 10., 100.,])  # wp/wc
# theta = np.array([[0.]*len(f), [45.]*len(f)]) / 180 * np.pi
theta = np.array([[0.]*len(f), [45.]*len(f), [80.]*len(f)]) / 180 * np.pi
# theta = np.array([[0.]*len(f), np.arccos(2*f), [45.]*len(f)])
# theta = np.array([[0.]*len(f), np.arccos(2*f), np.arccos(f)])

# %%
plt.figure(figsize=(20,5))
plt.suptitle('appleton hartree')
plt.subplot(1,3,1)
i = 0
for l in range(len(theta)):
    plt.plot(appleton_hartree(f, q[i], theta[l, :]), f, c='r', alpha=0.5)
    # plt.plot(appleton_hartree_(f, q[i], theta[l, :]), f, c='gray', alpha=0.5)
    plt.plot(appleton_hartree_quasi_parallel(f, q[i], theta[l, :]), f, c='b', alpha=0.5)
    # plt.plot(appleton_hartree_quasi_parallel_(f, q[i], theta[l, :]), f, c='b', alpha=0.5)
plt.xlim(0.1, 100.)
plt.ylim(0, 1.0)
plt.xscale('log')
plt.xlabel('ck/wc')
plt.ylabel('w/wc')
# plt.plot([-2, -1], [-2, -1], c='gray', alpha=0.5, label='without approximation l-mode')
plt.plot([-2, -1], [-2, -1], c='r', alpha=0.5, label='without approximation')
plt.plot([-2, -1], [-2, -1], c='b', alpha=0.5, label='with approximation')
plt.legend()
# plt.title('appleton hartree wp/wc='+str(q[i]))
plt.title('wp/wc='+str(q[i]))
plt.grid()
# plt.show()

plt.subplot(1,3,2)
i = 1
for l in range(len(theta)):
    plt.plot(appleton_hartree(f, q[i], theta[l, :]), f, c='r', alpha=0.5)
    # plt.plot(appleton_hartree_(f, q[i], theta[l, :]), f, c='gray', alpha=0.5)
    plt.plot(appleton_hartree_quasi_parallel(f, q[i], theta[l, :]), f, c='b', alpha=0.5)
    # plt.plot(appleton_hartree_quasi_parallel_(f, q[i], theta[l, :]), f, c='b', alpha=0.5)
plt.xlim(1., 1000.)
plt.ylim(0, 1.0)
plt.xscale('log')
plt.xlabel('ck/wc')
plt.ylabel('w/wc')
# plt.plot([-2, -1], [-2, -1], c='gray', alpha=0.5, label='without approximation l-mode')
plt.plot([-2, -1], [-2, -1], c='r', alpha=0.5, label='without approximation')
plt.plot([-2, -1], [-2, -1], c='b', alpha=0.5, label='with approximation')
plt.legend()
# plt.title('appleton hartree wp/wc='+str(q[i]))
plt.title('wp/wc='+str(q[i]))
plt.grid()
# plt.show()

plt.subplot(1,3,3)
i = 2
for l in range(len(theta)):
    plt.plot(appleton_hartree(f, q[i], theta[l, :]), f, c='r', alpha=0.5)
    # plt.plot(appleton_hartree_(f, q[i], theta[l, :]), f, c='gray', alpha=0.5)
    plt.plot(appleton_hartree_quasi_parallel(f, q[i], theta[l, :]), f, c='b', alpha=0.5)
    # plt.plot(appleton_hartree_quasi_parallel_(f, q[i], theta[l, :]), f, c='b', alpha=0.5)
plt.xlim(10., 10000.)
plt.ylim(0, 1.0)
plt.xscale('log')
plt.xlabel('ck/wc')
plt.ylabel('w/wc')
# plt.plot([-2, -1], [-2, -1], c='gray', alpha=0.5, label='without approximation l-mode')
plt.plot([-2, -1], [-2, -1], c='r', alpha=0.5, label='without approximation')
plt.plot([-2, -1], [-2, -1], c='b', alpha=0.5, label='with approximation')
plt.legend()
# plt.title('appleton hartree wp/wc='+str(q[i]))
plt.title('wp/wc='+str(q[i]))
plt.grid()
# plt.savefig('/Users/ampuku/Documents/duct/code/python/for_plot_omega_k_diagram/omega_k_diagram')
plt.savefig('/Users/ampuku/Documents/duct/code/python/for_plot_omega_k_diagram/omega_k_diagram1')
plt.show()

# %%
# q = np.arange(1.0, 10.0, 0.001)  # wp/wc
# f = np.arange(0., 1.0, 0.00001)  # w/wc
# idx_ck = np.arange(1.0, 10.0, 0.0001)*1e2  # wp/wc
# theta=45.
# deff = []
# for i, qq in enumerate(q):
#     dd = np.abs(appleton_hartree_quasi_parallel(f, qq, theta) - idx_ck[i])
#     ahw = np.double(appleton_hartree_quasi_parallel(f, qq, theta)[dd == min(dd)])
#     dd = np.abs(appleton_hartree(f, qq, theta) - idx_ck[i])
#     aho = np.double(appleton_hartree(f, qq, theta)[dd == min(dd)])
#     deff.append(abs(ahw - aho))
#     print(i)
# # %%
# plt.plot(q, deff)
# plt.xlabel('wp/wc')
# plt.ylabel('ck/wc (with approximation) - (without approximation)')

# %%
theta = np.array([0., 45.])
# theta = np.array([0., 45., 80.])
theta_ = theta / 180 * np.pi
# q = np.arange(1.0, 5.0, 0.001)  # wp/wc
q = np.arange(5.0, 15.0, 0.001)  # wp/wc
# q = np.array([0.1, 1., 10.])  # wp/wc
f = 0.4  # w/wc
for i in range(len(theta)):
    plt.plot(q, (np.sqrt(appleton_hartree(f, q, theta_[i])) - np.sqrt(appleton_hartree_quasi_parallel(f, q, theta_[i])))/np.sqrt(appleton_hartree_quasi_parallel(f, q, theta_[i]))*100, label='theta='+str(theta[i]))
plt.ylim(-0.01,0.3)
plt.xlabel('wp/wc')
plt.ylabel('wave number k err %')
plt.legend()
plt.grid()
plt.suptitle('appleton hartree quasi-parallel approximation')
plt.savefig('/Users/ampuku/Documents/duct/code/python/for_plot_omega_k_diagram/k_err')
# %%

# %%
# %%
theta = np.arange(0.0, 90.0, 0.01)
theta_ = theta / 180 * np.pi
q = 9.5  # wp/wc
# q = np.array([0.1, 1., 10.])  # wp/wc
f = [0.1,0.2,0.3,0.4]  # w/wc
plt.figure(figsize=(9,3))
for i in range(len(f)):
    plt.plot(theta, (np.sqrt(appleton_hartree(f[i], q, theta_)) - np.sqrt(appleton_hartree_quasi_parallel(f[i], q, theta_)))/np.sqrt(appleton_hartree_quasi_parallel(f[i], q, theta_))*100, label='f/fc='+str(f[i]))
plt.ylim(-0.01,1)
plt.xlabel('theta')
plt.ylabel('wave number k err %')
plt.legend()
plt.grid()
plt.suptitle('appleton hartree quasi-parallel approximation')
plt.savefig('/Users/ampuku/Documents/duct/code/python/for_plot_omega_k_diagram/k_err')

# %%
