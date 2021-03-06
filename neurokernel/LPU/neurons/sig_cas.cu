#define NNEU %(nneu)d

#define Kp 0.3
#define Kn 0.18
#define mp 2
#define mn 3
#define m 2
#define hM 40
#define hPLC 11.1
#define hD 37.8
#define hTpos 11.5
#define hTneg 10
#define ArateG 7.05
#define AratePLC 15.6
#define ArateT 150
#define ArateD 1300
#define ArateK 100
#define DrateM 3.7
#define DrateG 3.5
#define DratePLC 144
#define DrateD 4
#define DrateT 25
#define DrateGAP 3
#define percCa 0.40
#define Tcurrent 0.68
#define NaConcExt 120
#define NaConcInt 8
#define CaConcExt 1.5
#define CaConcInt 160
#define CaMConcInt 0.5
#define nBindingSites 4
#define FaradayConst 96485
#define AbsTemp 293
#define gasConst 8.314
#define CaUptakeRate 30
#define CaReleaseRate 5.5
#define CaDiffusionRate 1000
#define uVillusVolume 3.0*powf(10.0, -12.0)
#define NaCaConst 3.0*powf(10.0, -8.0)
#define membrCap 62.8
#define ns 1 //assuming a dim background (would be 2 for bright)
#define PLCT 100
#define GT 50
#define TT 25
#define CT 0.5

__global__ void signal_cascade(
    int neu_num,
    %(type)s *I_in,
    %(type)s *V_m,
    %(type)s *Np,
    %(type)s *rand1,
    %(type)s *rand2,
    %(type)s *Ca2,
    %(type)s *X_1,
    %(type)s *X_2,
    %(type)s *X_3,
    %(type)s *X_4,
    %(type)s *X_5,
    %(type)s *X_6,
    %(type)s *X_7,
    )
{

    for(int mid = 0; mid < 30000; ++mid){

    //16: state vector:
    double X1 = X_1[neu_num];
    double X2 = X_2[neu_num];
    double X3 = X_3[neu_num];
    double X4 = X_4[neu_num];
    double X5 = X_5[neu_num];
    double X6 = X_6[neu_num];
    double X7 = X_7[neu_num];
    //double X1 = Np[mid];
    //double X2 = G;
    //double X3 = activG;
    //double X4 = activPLC;
    //double X5 = activD;
    //double X6 = activC;
    //double X7 = activT;

    double Iin = Tcurrent*X7;

    double h[12];
    //18: reactant pairs - not concentrations??
    h[0] = X1;
    h[1] = X1*X2;
    h[2] = X3*(PLCT - X4);
    h[3] = X3*X4;
    h[4] = GT-X2-X3-X4;
    h[5] = X4;
    h[6] = X4; //NOT A TYPO
    h[7] = X5;
    h[8] = (X5*(X5-1)*(TT-X7))/2;
    h[9] = X7;
    h[10] = (CT - X6)*Ca2[neu_num];
    h[11] = X6;

    //31
    double fp = (powf((Ca2[neu_num]/Kp), mp)) / (1+powf((Ca2[neu_num]/Kp), mp));

    //32
    double fn = ns * powf((X6/Kn), mn)/(1+(powf((X6/Kn), mn)));

    double c[12];

    //19
    c[0] = DrateM * (1+hM*fn);

    //20
    c[1] = ArateG;

    //21
    c[2] = AratePLC;

    //22
    c[3] = DrateGAP;

    //23
    c[4] = DrateG;

    //24
    c[5] = ArateD;

    //25
    c[6] = DratePLC * (1+hPLC*fn);

    //26
    c[7] = DrateD*(1+hD*fn);

    //27
    c[8] = (ArateT*(1+hTpos*fp))/(ArateK*ArateK);

    //28
    c[9] = DrateT*(1+hTneg*fn);

    //29
    c[10] = CaUptakeRate/(uVillusVolume*uVillusVolume);

    //30
    c[11] = CaReleaseRate;

    //need an a vector:
    double a0 = c[0]*h[0];
    double a1 = c[1]*h[1];
    double a2 = c[2]*h[2];
    double a3 = c[3]*h[3];
    double a4 = c[4]*h[4];
    double a5 = c[5]*h[5];
    double a6 = c[6]*h[6];
    double a7 = c[7]*h[7];
    double a8 = c[8]*h[8];
    double a9 = c[9]*h[9];
    double a10 = c[10]*h[10];
    double a11 = c[11]*h[11];

    double as = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11;

    double av[12];
    av[0] = h[0]*c[0];
    double hc[12];
    hc[0] = av[0];
    int mu = 0;
    // 12 possible reaction
    for(int ii = 1; ii < 12; ++ii){
        hc[ii] = c[ii]*h[ii];
        av[ii] = av[ii-1] + hc[ii];

        if((rand2[mid]*as > av[ii - 1]) && (rand2[mid]*as <= av[ii])){
            mu = ii;
        }
    }

    if(mu == 0) {
        X_1[neu_num] += -hc[mu];
    } else if (mu == 1){
        X_2[neu_num] += -hc[mu];
        X_3[neu_num] += hc[mu];
    } else if (mu == 2){
        X_3[neu_num] += -hc[mu];
        X_4[neu_num] += hc[mu];
    } else if (mu == 3){
        X_3[neu_num] += -hc[mu];
    } else if (mu == 4){
        X_2[neu_num] += hc[mu];
    } else if (mu == 5){
        X_5[neu_num] += hc[mu];
    } else if (mu == 6){
        X_4[neu_num] += -hc[mu];
    } else if (mu == 7){
        X_5[neu_num] += -hc[mu];
    } else if (mu == 8){
        X_5[neu_num] += -2 * hc[mu];
        X_7[neu_num] += hc[mu];
    } else if (mu == 9){
        X_7[neu_num] += -hc[mu];
    } else if (mu == 10){
        X_6[neu_num] += hc[mu];
    } else {
        X_6[neu_num] += -hc[mu];
    }

    I_in[mid] = Tcurrent*X_7[neu_num];

    }
}
