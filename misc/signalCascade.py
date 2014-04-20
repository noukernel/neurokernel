import pycuda.autoinit 
import pycuda.driver as drv 
import numpy 
 
from pycuda.compiler import SourceModule 
import random 
 
mod = SourceModule(""" 

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

__global__ void signalCascade(
    int *I_in,
    double *V_m,
    double *Np,
    double *rand1,
    double *rand2,
    double *Ca2)
{

for(int mid = 0; mid < 30000; ++mid){

//intial conditions given as zero for almost everything:
double G = 50;
double activG = 0;
double activPLC = 0;
double activD = 0;
double activC = 0;
double activT = 0;

double Iin = Tcurrent*activT;

//16: state vector:
double X1 = Np[mid];
double X2 = G;
double X3 = activG;
double X4 = activPLC;
double X5 = activD;
double X6 = activC;
double X7 = activT;

//17: state transition matrix, sparsely defined
int V11 = -1;
int V22 = -1;
int V25 = 1;
int V32 = 1;
int V33 = -1;
int V34 = -1;
int V43 = 1;
int V47 = -1;
int V56 = 1;
int V58 = -1;
int V59 = -2;
int V611 = 1;
int V612 = -1;
int V79 = 1;
int V710 = -1;


double h[12];
//18: reactant pairs - not concentrations??
h[0] = Np[mid];
h[1] = Np[mid]*G;
h[2] = activG*(PLCT - activPLC);
h[3] = activG*activPLC;
h[4] = GT-activG-G-activPLC;
h[5] = activPLC;
h[6] = activPLC; //NOT A TYPO
h[7] = activD;
h[8] = (activD*(activD-1)*(TT-activT))/2;
h[9] = activT;
h[10] = (CT - activC)*Ca2;
h[11] = activC;

double c[12];
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

//31
double fp = (powf((Ca2/Kp), mp)) / (1+powf((Ca2/Kp), mp));

//27
c[8] = (ArateT*(1+hTpos*fp))/(ArateK*ArateK);

//32
double fn = ns * powf((activC/Kn), mn)/(1+(powf((activC/Kn), mn)));
//might be a problem wtih activC vs activeCint not being the same thing

//19
c[0] = DrateM * (1+hM*fn);

//25
c[6] = DratePLC * (1+hPLC*fn);

//26
c[7] = DrateD*(1+hD*fn);

//28
c[9] = DrateT*(1+hTneg*fn);

//29
c[10] = CaUptakeRate/(uVillusVolume*uVillusVolume);

//30
c[11] = CaReleaseRate;

//need an a vector:
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
double a12 = c[12]*h[12];

double as = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12;

double av[12];
av[0] = 0;
double hc[12];
int mu = 0;
// 12 possible reaction
for(int ii = 1; ii < 12; ++ii){
    hc[ii] = c[ii]*h[ii];
    av[ii] += hc[ii];

    if((rand2[mid]*as > av[ii - 1]) && (rand2[mid]*as <= av[ii])){
        mu = ii;
    }
}

if(mu == 0) {
    X1 += -hc[mu];
} else if (mu == 1){
    X2 += -hc[mu];
    X3 += hc[mu];
} else if (mu == 2){
    X3 += -hc[mu];
    X4 += hc[mu];
} else if (mu == 3){
    X3 += -hc[mu];
} else if (mu == 4){
    X2 += hc[mu];
} else if (mu == 5){
    X5 += hc[mu];
} else if (mu == 6){
    X4 += -hc[mu];
} else if (mu == 7){
    X5 += -hc[mu];
} else if (mu == 8){
    X5 += -2 * hc[mu];
    X7 += hc[mu];
} else if (mu == 9){
    X7 += -hc[mu];
} else if (mu == 10){
    X6 += hc[mu];
} else {
    X6 += -hc[mu];
}

I_in[mid] = mu;

}
}
}

""", options = ["--ptxas-options=-v"])

Np = numpy.zeros(30000)
rand1 = numpy.random.uniform(size=30000)
rand2 = numpy.random.uniform(size=30000)
I_in = numpy.zeros(30000, dtype=numpy.int32)
V_m = numpy.ones(30000) * -0.07

signalCascade = mod.get_function("signalCascade")
signalCascade(drv.Out(I_in), drv.In(V_m), drv.In(Np), drv.In(rand1), drv.In(rand2), block=(1,1,1), grid=(1,1))

print I_in
