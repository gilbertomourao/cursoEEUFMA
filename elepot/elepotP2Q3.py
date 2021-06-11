from xnlpy import *

VLL = 220
R = 15

vab = lambda wt: sqrt(2)*VLL*sin(wt + pi/6)
vac = lambda wt: sqrt(2)*VLL*sin(wt - pi/6)
vba = lambda wt: sqrt(2)*VLL*sin(wt - 150 * pi / 180)
vcb = lambda wt: sqrt(2)*VLL*sin(wt + pi/2)

alpha = 25 * pi / 180

# Cálculo de Isef

isrc = lambda wt, a: (1/R) * (((wt >= pi/6+a) & (wt < pi/2+a)) * vab(wt) + 
							  ((wt >= pi/2+a) & (wt < 5*pi/6+a)) * vac(wt) - 
							  ((wt >= 7*pi/6+a) & (wt < 9*pi/6+a)) * vba(wt) - 
							  ((wt >= 9*pi/6+a) & (wt < 11*pi/6+a)) * vcb(wt))

Isef = sqrt(1/(2*pi) * integral(lambda wt,a: isrc(wt,a)**2, alpha, 2*pi+alpha, args=(alpha,), depth=40)[0])

print("Isef = %.2f A" % Isef)

# Componente fundamental da tensão

a1 = (1/pi) * integral(lambda wt,a: R*isrc(wt,a)*cos(wt), alpha, 2*pi+alpha, args=(alpha,), depth=40)[0]
b1 = (1/pi) * integral(lambda wt,a: R*isrc(wt,a)*sin(wt), alpha, 2*pi+alpha, args=(alpha,), depth=40)[0]

print("a1 = %.2f\nb1 = %.2f" % (a1,b1))

is1 = lambda wt: (a1*sin(wt) + b1*cos(wt))/R

Is1ef = sqrt((1/(2*pi) * integral(lambda wt: is1(wt)**2, alpha, 2*pi + alpha)[0]))

print("Is1ef = %.2f A" % Is1ef)

THD = 100 * sqrt((Isef/Is1ef)**2 - 1)

print("THDi = %.2f%%" % THD)