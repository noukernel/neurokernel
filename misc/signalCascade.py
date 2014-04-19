import pycuda.autoinit 
import pycuda.driver as drv 
import numpy 
 
from pycuda.compiler import SourceModule 
import random 
 
mod = SourceModule(""" 

#define posCoef 0.3
#define negCoef 0.18
#define posConst 2
#define negConst 3
#define m 2
#define hM 40
#define hPLC 11.1
#define hD 37.8
#define hTpos 11.5
#define hTneg 10
#define ArateG 7.05
#define AratePLC 15.6
#define ArateT 150
#define ArateD 100
#define DrateM 3.7
#define DrateG 3.5
#define DratePLC 144
#define DrateD 4
#define DrateT 25
#define DrateGAP 3
#define Tchannels 25
#define numGprotein 50
#define numPLC 100
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
#define GasConst 8.314
#define CaUptakeRate 30
#define CaReleaseRate 5.5
#define CaDiffusionRate 1000
#define uVillusVolume 3.0*double pow(double 10, double -12)
#define NaCaConst 3.0*double pow(double 10, double -8)
#define membrCap 62.8

__global__ void signalCascade{

double Vm = -0.070;
double Iin = Tcurrent*activT;

//16: state vector:
double X1 = activM;
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


//18: reactant pairs - not concentrations??
double h1 = activM;
double h2 = activM*G;
double h3 = activG*(PLC - activPLC);
double h4 = activG*activPLC;
double h5 = GT-activG-G-activPLC;
double h6 = activPLC;
double h7 = activPLC; //NOT A TYPO
double h8 = activD;
double h9 = (activD*(activD-1)*(T-activT))/2;
double h10 = activT;
double h11 = CaCaM;
double h12 = activC;

//20
double c2 = ArateG;

//21
double c3 = AratePLC;

//22
double c4 = DrateGAP;

//23
double c5 = DrateG;

//24
double c6 = ArateD;

//31
double posFeedback = (pow(CaConcInt/posCoef), posConst)/(1+(pow(CaConcInt/posCoef), posConst));

//27
double c9 = (ArateT*(1+hTpos*posFeedback))/(ArateD*ArateD);

//32
double negFeedback = ns * (pow(ActiveCint/negCoef), negConst)/(1+(pow(ActiveCint/negCoef), negConst));

//19
double c1 = DrateM * (1+hM*negFeedback);

//25
double c7 = DratePLC * (1+hPLC*negFeedback);

//26
double c8 = DrateD*(1+hD*negFeedback);

//28
double c10 = DrateT*(1+hTneg*negFeedback);

//29
double c11 = CaUptakeRate/(uVillusVolume*uVillusVolume);

//30
double c12 = CaReleaseRate;

//33 and 34 are about timestep choice

//35
double ddtCaInt = netCaCurrent/(2 * uVillusVolume * FaradayConst)-n*ddtActiveCint - CaDiffusionRate*CaConcInt;

double CaCurrent = Iin * percCa;
double NaCaCurrent = NaCaConst *(pow(NaConcInt,3)*CaConcExt-pow(NaConcExt,3.0)*CaConcInt*exp((Vm*FaradayConst)/(gasConst*AbsTemp));

//36
double netCaCurrent = CaCurrent - 2*NaCaCurrent;

double ActiveCconc = ?;    

//41
double f1 = NaCaConst * pow(NaConcInt, 3.0)*pow(CaConcExt, 2.0) / (uVillusVolume * FaradayConst);
//42
double f2 = NaCaConst * exp((-Vm*FaradayConst)/(gasConst*AbsTemp)) * pow(NaConcExt,3.0) / (uVillusVolume * FaradayConst);

//40 (composed of 37,38,39)
double num = netCaCurrent/(2*uVillusVolume * FaradayConst)+n*CaReleaseRate*activCconc - f1;
double den = n*CaUptakeRate*CaMConcInt+CaDiffusionRate-f2;
double steadyStateCa = uVillusVolume*(num/den);

}

""", options = ["--ptxas-options=-v"])





signalCascade = mod.get_function("signalCascade")
signalCascade(block=(1,1,1), grid=(1,1))









