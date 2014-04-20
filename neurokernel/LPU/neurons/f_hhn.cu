#define NNEU %(nneu)d
#define E_K -85         // Potassium Reverse Potential
#define E_cl -30        // Chlorine Reverse Potential
#define G_s 1.6         // Maximum Shaker Conductance
#define G_dr 3.5        // Maximum Delayed Rectifier Conductance
#define G_Cl 0.056      // Chlorine Leak Conductance
#define G_K 0.082       // Potassium Leak Conductance
#define C = 4           // Membrane Capacitance

__global__ f_hhn(
    %(type)s *I,
    %(type)s *V,
    %(type)s *SA,
    %(type)s *SI,
    %(type)s *DRA,
    %(type)s *DRI
    )
{   
    %(type)s i_ext = I;
    %(type)s v = V;
    %(type)s sa = SA;
    %(type)s si = SI;
    %(type)s dra = DRA;
    %(type)s dri = DRI;
    float sa_inf = powf(1/(1 + exp((-30-v)/13.5)), (1/3));
    float sa_tau = 0.13+3.39*exp(-powf(-73-v, 2)/pow(20,2));
    float si_inf = 1/(1 + exp((-55-v)/-5.5));
    float si_tau = 113 * exp(-pow(-71-v, 2)/pow(29,2));
    float dra_inf = powf(1/(1 + exp((-5-v))), 0.5);
    float dra_tau = 0.5 + 5.75 * exp(-pow(-25-v, 2)/pow(32,2));
    float dri_inf = 1 / (1 + exp((-25-v) / -10.5));
    float dri_tau = 890;

    //Runge-Kutta?
}
