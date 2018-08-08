import numpy as np
import pdb
import scipy.constants as ct
from scipy.special import ellipk as EllipticK
from scipy.special import ellipe as EllipticE
from mpmath import ellippi

Cos = np.cos
Pi = np.pi
Sqrt = np.sqrt
Msun = 1.989e30

def EllipticPi(n, m):
	return float(ellippi(n, Pi/2., m))

def KerrGeoFreqs_vectorized(a, p, e, theta_inc, M=1):
	KGF_vec = np.frompyfunc(KerrGeoFreqs_scalar, 4, 4)
	#pdb.set_trace()
	return KGF_vec(a, p, e, theta_inc)

def convert_to_Hz(Mf, M):
	M_time = M*Msun*ct.G/ct.c**3
	return Mf/M

def KerrGeoELQ(a, p, e, theta_inc):
	"""
	KerrGeoELQ[a_(*/;Abs[a]<=1*), p_, e_, \[Theta]inc1_?NumericQ] := Module[{M=1,f, g, h, d, fp, gp, hp, dp, r, rp, ra, zm, \[CapitalDelta], \[Rho], \[Kappa], \[Epsilon], \[Eta], \[Sigma], En, L, Q, E1, Em1, f1, g1, h1, d1, f2, g2, h2, d2, L1, L2,r0,\[CapitalDelta]0,Z,\[Theta]min,\[Theta]inc=\[Theta]inc1},

	
	\[Theta]inc=Mod[\[Theta]inc,2\[Pi]];
	If[\[Theta]inc>\[Pi], \[Theta]inc=2\[Pi]-\[Theta]inc];
	If[\[Theta]inc <= \[Pi]/2 , \[Theta]min = \[Pi]/2-\[Theta]inc, \[Theta]min=-\[Pi]/2+\[Theta]inc];

	If[Mod[\[Theta]inc,\[Pi]]==\[Pi]/2 && e!=0,Print["Polar, non-spherical orbits not yet implemented"]; Return[]];

	"""
	if theta_inc <= Pi/2.:
		theta_min = Pi/2. - theta_inc

	elif theta_inc != Pi/2.:
		theta_min = Pi/2. + theta_inc

	else:
		raise Exception("Polar, non-spherical orbits not yet implemented")


	rp = p/(1 + e)
	ra = p/(1 - e)

	zm = Cos(theta_min)
 
 
	Delta_r = lambda r : r**2 - 2*r + a**2
 
 
	f = lambda r : r**4 + a**2*(r*(r+2) + zm**2 * Delta_r(r))
	g = lambda r : 2*a*r
	h = lambda r : r*(r - 2) + zm**2/(1 - zm**2) * Delta_r(r)
	d = lambda r : (r**2 + a**2 * zm**2) * Delta_r(r)
 
	fp = lambda r : 4*r**3 + 2*a**2 ((1 + zm**2)*r + (1 - zm**2))
	gp = 2*a
	hp = lambda r : 2*(r - 1)/(1 - zm**2)
	dp = lambda r : 2*(2*r - 3) * r**2 + 2*a**2 ((1 + zm**2)*r - zm**2)
 
	f1, g1, h1, d1 = f(rp), g(rp), h(rp), d(rp)
 
	if e != 0:
		f2, g2, h2, d2 = f(ra), g(ra), h(ra), d(ra)
	else:
		f2, g2, h2, d2 = fp(ra), gp(ra), hp(ra), dp(ra)
 
	Kappa = d1*h2 - h1*d2
	Epsilon = d1*g2 - g1*d2
	Rho = f1*h2 - h1*f2
	Eta = f1*g2 - g1*f2
	Sigma = g1*h2 - h1*g2
 
	pm = 1
	En = Sqrt((Kappa*Rho + 2*Epsilon * Sigma - pm*2*Sqrt(Sigma*(Sigma*Epsilon**2 + Rho*Epsilon*Kappa - Eta * Kappa**2)))/(Rho**2 + 4*Eta*Sigma))

	L = -(En*g1/h1) + pm*Sqrt((g1*En/h1)**2 + (f1*En**2 - d1)/h1)

	Q = zm**2 * (a**2 * (1 - En**2) + L**2/(1 - zm**2))

	return En, L, Q

def KerrGeoRadialRoots(a, p, e, theta_inc, M=1):
	En,L,Q=KerrGeoELQ(a,p,e,theta_inc)

	r1=p/(1-e)
	r2=p/(1+e)
	AplusB=(2*M)/(1-En**2)-(r1+r2) #(*Eq. (11)*)
	AB=(a**2 * Q)/((1-En**2)*r1 * r2) #(*Eq. (11)*)
	r3=(AplusB+Sqrt((AplusB)**2-4*AB))/2. #(*Eq. (11)*)
	r4=AB/r3

	return r1,r2,r3,r4

def KerrGeoPolarRoots(a,p,e, theta_inc):
  En,L,Q = KerrGeoELQ(a, p, e, theta_inc)
  theta_min=(Pi/2-theta_inc)/np.sign(L)
  zm = Cos(theta_min)
  zp = (a**2 * (1-En**2)+L**2/(1-zm**2))**(1/2)
  return zp,zm


def KerrGeoFreqs_scalar(a, p, e, theta_inc, M=1):
	"""
	KerrGeoFreqs[a_/;Abs[a]<1,p_,e_,\[Theta]inc1_?NumericQ]:=Module[{M=1,En,L,Q,r1,r2,AplusB,AB,r3,r4,\[Epsilon]0,zm,kr,k\[Theta],Upsilon_r,Upsilon_theta,\[CapitalUpsilon]\[Phi],\[CapitalGamma],rp,rm,hp,hm,hr,EnLQ,a2zp,epsilon_0zp,zmOverZp,\[Theta]min,\[Theta]inc=\[Theta]inc1},
	"""
	#theta_inc=Mod[\[Theta]inc,2\[Pi]];
	if theta_inc > Pi:
		theta_inc = 2*Pi-theta_inc

	if theta_inc == Pi/2:
		raise Exception("Equations for polar orbits not implemented yet")
	

	En,L,Q=KerrGeoELQ(a,p,e,theta_inc)

	theta_min=(Pi/2-theta_inc)/np.sign(L);

	r1,r2,r3,r4 = KerrGeoRadialRoots(a,p,e,theta_inc)

	
	epsilon_0= a**2 * (1-En**2)/L**2
	zm=Cos(theta_min)**2
	a2zp=(L**2+a**2 * (-1+En**2) * (-1+zm))/((-1+En**2) *(-1+zm))

	epsilon_0zp=-((L**2+a**2 * (-1+En**2) * (-1+zm))/(L**2 * (-1+zm)))

	if a == 0.0:
		zmOverZp = 0.0
	else:
		zmOverZp = zm/((L**2+a**2 * (-1+En**2) * (-1+zm))/(a**2 * (-1+En**2) * (-1+zm)))

	kr=Sqrt((r1-r2)/(r1-r3)*(r3-r4)/(r2-r4)) #(*Eq.(13)*)
	ktheta =Sqrt(zmOverZp) #(*Eq.(13)*)
	Upsilon_r =(Pi * Sqrt((1-En**2)*(r1-r3)*(r2-r4)))/(2*EllipticK(kr**2)) #(*Eq.(15)*)
	Upsilon_theta =(Pi * L * Sqrt(
		epsilon_0zp))/(2*EllipticK(ktheta**2)) #(*Eq.(15)*)

	rp=M+Sqrt(M**2-a**2)
	rm=M-Sqrt(M**2-a**2)
	hr=(r1-r2)/(r1-r3)
	hp=((r1-r2)*(r3-rp))/((r1-r3)*(r2-rp))
	hm=((r1-r2)*(r3-rm))/((r1-r3)*(r2-rm))

	Upsilon_phi=(2*Upsilon_theta)/(Pi * Sqrt(epsilon_0zp)) * EllipticPi(zm,ktheta**2)+(2*a* Upsilon_r)/(Pi*(rp-rm)*Sqrt((1-En**2)*(r1-r3)*(r2-r4))) * ((2* M * En * rp - a * L)/(r3-rp) * (EllipticK(kr**2)-(r2-r3)/(r2-rp) * EllipticPi(hp,kr**2))-(2*M * En * rm - a * L)/(r3-rm) * (EllipticK(kr**2)-(r2-r3)/(r2-rm) * EllipticPi(hm,kr**2))) #(*Eq. (21)*)


	#(*Convert to frequencies w.r.t BL time using Fujita and Hikida's formula Eq. (21)*)
	Gamma = 4*M**2 * En + (2*a2zp * En * Upsilon_theta)/(Pi * L * Sqrt(epsilon_0zp)) * (EllipticK(ktheta**2)- EllipticE(ktheta**2)) + (2*Upsilon_r)/(Pi* Sqrt((1-En**2)*(r1-r3)*(r2-r4))) * (En/2 * ((r3*(r1+r2+r3)-r1 * r2)*EllipticK(kr**2)+(r2-r3)*(r1+r2+r3+r4)*EllipticPi(hr,kr**2)+(r1-r3)*(r2-r4)*EllipticE(kr**2))+2*M * En*(r3 * EllipticK(kr**2)+(r2-r3)*EllipticPi(hr,kr**2))+(2*M)/(rp-rm) * (((4*M**2 * En-a * L)*rp-2*M* a**2 * En)/(r3-rp) * (EllipticK(kr**2)-(r2-r3)/(r2-rp) *EllipticPi(hp,kr**2))-((4*M**2 * En-a * L)*rm-2*M * a**2 * En)/(r3-rm) * (EllipticK(kr**2)-(r2-r3)/(r2-rm) * EllipticPi(hm,kr**2))))

	#(*Output the BL frequencies by dividing the Mino time frequencies by the conversion factor \[CapitalGamma]*)
	return Upsilon_r/Gamma, np.fabs(Upsilon_theta/Gamma),Upsilon_phi/Gamma, Gamma

def KerrGeoFreqs(a, p, e, theta_inc, M=1):
	return KerrGeoFreqs_vectorized(a, p, e, theta_inc)


if __name__ == "__main__":

	mu = 10.0
	M = 1e6*1.989e30*ct.G/ct.c**2
	num = 100
	a = np.full(num,0.44)
	p = np.full(num, 7.7) #really p/M
	e = np.full(num, 0.7)
	iota = np.full(num, np.pi/5.)

	a = 0.732
	p = 8.7
	e = 0.6
	iota = Pi/2.5
	

	import time
	st = time.time()
	check = KerrGeoFreqs_vectorized(a, p, e, iota)
	print(time.time()-st)
	print(check)
	pdb.set_trace()
