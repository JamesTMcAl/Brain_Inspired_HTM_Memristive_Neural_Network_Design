.title kwta
.OPTIONS POST=1 LIST ingold=2 runlvl=0


.subckt Invert in out
Vdd N_Vdd 0 5
M10 out in N_Vdd N_Vdd PMOD W=100n L=100n
M11 out in 0 0 NMOD W=100n L=100n
.ends Invert

****************************************************
.subckt Mod in out v1 v1_ m16_s v3 itail
Vdd N_Vdd 0 5
Vcc N_Vcc 0 -5

M11 in v1 m10_s N_Vdd PMOD W=4u L=5u
M10 in v1_ m10_s N_Vcc NMOD W=4u L=5u
X1 m10_s 1 Invert
X2 1 out Invert
M5 N_Vdd m16_s m5_s N_Vdd PMOD w=7u l=4u
M7 m5_s m3_s m10_s N_Vdd PMOD w=7u l=4u
M8 m10_s itail m8_s N_Vcc nmod w=13u l=5u
M9 m10_s v3 m8_s N_Vcc nmod w=13u l=5u
M6 m8_s m2_s 0 N_Vcc nmod w=11u l=5u
M3 N_Vdd m16_s m3_s N_Vdd pmod w=21u l=4u
M1 m3_s m10_s itail N_Vcc nmod w=9u l=5u
M2 m3_s itail m2_s N_Vdd pmod w=17u l=4u
M4 m2_s m2_s 0 N_Vcc nmod w=11u l=5u
.ends Mod
****************************************************

****************************************************
.subckt Cursour v2 i1 i2 itail m16_s
Vdd N_Vdd 0 5
Vcc N_Vcc 0 -5

M16 N_Vdd m16_s m16_s N_Vdd pmod w=21u l=4u
M17 N_Vdd v2 m16_s N_Vdd pmod w=31u l=5u
M13 itail i2 0 N_Vcc nmod w=31u l=5u
M12 i2 i2 0 N_Vcc nmod w=31u l=5u
M15 m16_s i1 0 N_Vcc nmod w=31u l=5u
M14 i1 i1 0 N_Vcc nmod w=31u l=5u
.ends Cursour
****************************************************

V1 N_v1 0 PWL 0 0 2u 0 2.01u 5 4u 5 4.01u 5
V2 N_v2 0 PWL 0 0 4u 0 4.01u 5 10u 5 10.01u 5
V3 N_v3 0 PWL 0 0 2u 0 14u 0 14.01u 5 16u 5
*V1_ N_v1_ 0 PWL 0 5 2u 5 2.01u 0
I1 N_i1 0 20u
I2 N_i2 0 40u

Xinvert N_v1 N_v1_ Invert

Xcursour N_v2 N_i1 N_i2 N_itail N_m16s Cursour

Xm1 N_vin1 N_vout1 N_v1 N_v1_ N_m16s N_v3 N_itail Mod
Xm2 N_vin2 N_vout2 N_v1 N_v1_ N_m16s N_v3 N_itail Mod
Xm3 N_vin3 N_vout3 N_v1 N_v1_ N_m16s N_v3 N_itail Mod
*Xm4 N_vin4 N_vout4 N_v1 N_v1_ N_m16s N_v3 N_itail Mod
*Xm5 N_vin5 N_vout5 N_v1 N_v1_ N_m16s N_v3 N_itail Mod
*Xm6 N_vin6 N_vout6 N_v1 N_v1_ N_m16s N_v3 N_itail Mod

Vin1 N_vin1 0 0.1
Vin2 N_vin2 0 0.2
Vin3 N_vin3 0 0.3
*Vin4 N_vin4 0 0.4
*Vin5 N_vin5 0 0.5
*Vin6 N_vin6 0 0.6


.MODEL NMOD NMOS LEVEL=1
.MODEL PMOD PMOS LEVEL=1

.op
.tran 0.01us 100us UIC
*.tran 0.01us 100us
.option post accurate probe

.probe tran v(r) V(N_vout1) V(N_vout2) V(N_vout3) V(N_vout4) V(N_vout5) V(N_vout6) 
+ V(N_v1) V(N_v1_) V(N_v2) V(N_v3)
+ par('V(Xm1.m10_s)') par('V(Xm2.m10_s)') par('V(Xm3.m10_s)') par('V(Xm4.m10_s)') par('V(Xm5.m10_s)') par('V(Xm6.m10_s)')

.end