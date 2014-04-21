#define NNEU %(nneu)d
#define E0 -77
#define E1 50
#define E2 -54.387
#define g0 36
#define g1 120
#define g2 0.3

__global__ void hodgkin_huxley(
    int neu_num,
    %(type)s dt,
    %(type)s *V,
    %(type)s *I,
    %(type)s *X0,
    %(type)s *X1,
    %(type)s *X2)
{
    int bid = blockIdx.x;
    int nid = bid * NNEU + threadIdx.x;

    %(type)s v, i_ext, x0, x1, x2;

    if( nid < neu_num ){
        v = V[nid];
        i_ext = I[nid];
        x0 = X0[nid];
        x1 = X1[nid];
        x2 = X2[nid];

        v = v + ( ( i_ext - (g1*pow(x1,3)*x2*(v-50)) - (g0*pow(x0, 4)*(v+77)) - (g2*(v+54.387))) * dt);

        %(type)s a;
        a = exp(-(v+55)/10) - 1;
        if (a == 0){
            x0 = x0+((((1 - x0) * 0.1) - (x0 * 0.125 * exp(-(v+65)/80))) * dt);
        } else {
            x0 = x0+(( (1-x0) * (-0.01*(v+55)/a) - (x0 * (0.125 * exp(-(v+65)/80))) )*dt);
        }

        a = exp(-(v+40)/10)-1;
        if (a == 0){
            x1 = x1 + (( (1-x1) - (x1*(4*exp(-(v+65)/18)))) * dt);
        } else {
            x1 = x1 + (( ((1-x1) * (-0.1*(v+40)/a)) - (x1 * (4 * exp(-(v+65)/18))) ) *dt);
        }

        x2 = x2 + (( ((1 - x2)* 0.07*exp(-(v+65)/20)) - (x2 / (exp(-(v+35)/10) + 1)) ) * dt);

        V[nid] = v;
        X0[nid] = x0;
        X1[nid] = x1;
        X2[nid] = x2;
    }
    return;
}
