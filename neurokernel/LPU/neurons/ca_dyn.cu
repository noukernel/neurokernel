#define NNEU %(nneu)d
#define Kp 0.3
#define Kn 0.18
#define mp 2
#define mn 3
#define Na_o 120
#define Na_i 8
#define Ca_o 1.5
#define Ca_id 160.0*powf(10.0, -6.0)
#define P_Ca 0.4
#define K_NaCa 3.0*powf(10.0, -8.0)
#define F 96485
#define R 8.314
#define T 293
#define v 3.0*powf(10.0, -12.0)
#define n 4
#define K_r 5.5
#define K_u 30
#define C_T 0.5
#define K_Ca 100

__global__ void calciumDynamics(
    int neu_num,
	%(type)s *Ca2,
	%(type)s *V_m,
	%(type)s *I_in,
	%(type)s *C_star)
{

    double I_Ca = I_in[neu_num] * P_Ca;
    double I_NaCa = K_NaCa * (powf(Na_i,3.0) * Ca_o - powf(Na_o,3.0) * Ca2[neu_num] * exp((-V_m[neu_num]*F) / (R*T)));

    //36
    double I_CaNet = I_Ca - 2*I_NaCa;

    //41
    double f1 = K_NaCa * powf(Na_i, 3.0)*Ca_o / (v*F);
    //42
    double f2 = (K_NaCa * exp((-V_m[neu_num]*F)/(R*T)) * powf(Na_o,3.0))/(v*F);

    //40 (composed of 37,38,39)
    Ca2[neu_num] = v*(I_CaNet/(2*v*F) + n*K_r*C_star[neu_num] - f1)/(n*K_u*(C_T - C_star[neu_num]) + K_Ca - f2);

}
