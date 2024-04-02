#include "incompressible_navier_stokes_q1_p0_structured_element.hpp"

void IncompressibleNavierStokesQ1P0StructuredElement::CalculateRightHandSide(
    const double a,
    const double b,
    const double mu,
    const double rho,
    const Eigen::Array<double, 4, 2>& v,
    const double p,
    const Eigen::Array<double, 4, 2>& f,
    const Eigen::Array<double, 4, 2>& acc,
    Eigen::Array<double, 8, 1>& RHS)
{

    const double cRHS0 = pow(a, -2);
    const double cRHS1 = 0.62200846792814624*v(0,0);
    const double cRHS2 = 0.16666666666666669*v(3,0);
    const double cRHS3 = -0.16666666666666669*v(2,0);
    const double cRHS4 = cRHS2 + cRHS3;
    const double cRHS5 = cRHS1 + cRHS4 - 0.62200846792814624*v(1,0);
    const double cRHS6 = pow(b, -2);
    const double cRHS7 = 0.044658198738520456*v(0,0);
    const double cRHS8 = 0.044658198738520456*v(3,0);
    const double cRHS9 = 0.16666666666666669*v(1,0);
    const double cRHS10 = cRHS3 + cRHS9;
    const double cRHS11 = cRHS6*(-cRHS10 - cRHS7 + cRHS8);
    const double cRHS12 = cRHS4 + cRHS7 - 0.044658198738520456*v(1,0);
    const double cRHS13 = 0.62200846792814624*v(3,0);
    const double cRHS14 = cRHS6*(-cRHS1 - cRHS10 + cRHS13);
    const double cRHS15 = 0.78867513459481287*v(3,0);
    const double cRHS16 = 0.21132486540518713*v(1,0);
    const double cRHS17 = 0.21132486540518713*v(0,0) - 0.78867513459481287*v(2,0);
    const double cRHS18 = cRHS15 - cRHS16 + cRHS17;
    const double cRHS19 = -cRHS18;
    const double cRHS20 = cRHS0*cRHS19;
    const double cRHS21 = 0.78867513459481287*v(1,0);
    const double cRHS22 = 0.21132486540518713*v(3,0);
    const double cRHS23 = cRHS17 + cRHS21 - cRHS22;
    const double cRHS24 = -cRHS23;
    const double cRHS25 = cRHS24*cRHS6;
    const double cRHS26 = 0.21132486540518713*mu;
    const double cRHS27 = 0.78867513459481287*v(0,0) - 0.21132486540518713*v(2,0);
    const double cRHS28 = -cRHS21 + cRHS22 + cRHS27;
    const double cRHS29 = -cRHS28;
    const double cRHS30 = cRHS0*cRHS29;
    const double cRHS31 = -cRHS15 + cRHS16 + cRHS27;
    const double cRHS32 = -cRHS31;
    const double cRHS33 = cRHS32*cRHS6;
    const double cRHS34 = 0.78867513459481287*mu;
    const double cRHS35 = 0.16666666666666669*f(1,0) + 0.16666666666666669*f(3,0);
    const double cRHS36 = rho*(cRHS35 + 0.62200846792814624*f(0,0) + 0.044658198738520456*f(2,0));
    const double cRHS37 = rho*(cRHS35 + 0.044658198738520456*f(0,0) + 0.62200846792814624*f(2,0));
    const double cRHS38 = 1.0/a;
    const double cRHS39 = 2.0*p;
    const double cRHS40 = cRHS38*cRHS39;
    const double cRHS41 = -cRHS40;
    const double cRHS42 = 0.16666666666666669*v_conv(0,0) + 0.16666666666666669*v_conv(2,0);
    const double cRHS43 = cRHS38*(cRHS42 + 0.044658198738520456*v_conv(1,0) + 0.62200846792814624*v_conv(3,0));
    const double cRHS44 = 1.0/b;
    const double cRHS45 = 0.16666666666666669*v_conv(0,1) + 0.16666666666666669*v_conv(2,1);
    const double cRHS46 = cRHS44*(cRHS45 + 0.044658198738520456*v_conv(1,1) + 0.62200846792814624*v_conv(3,1));
    const double cRHS47 = cRHS19*cRHS43 + cRHS32*cRHS46;
    const double cRHS48 = cRHS38*(cRHS42 + 0.62200846792814624*v_conv(1,0) + 0.044658198738520456*v_conv(3,0));
    const double cRHS49 = cRHS44*(cRHS45 + 0.62200846792814624*v_conv(1,1) + 0.044658198738520456*v_conv(3,1));
    const double cRHS50 = cRHS24*cRHS49 + cRHS29*cRHS48;
    const double cRHS51 = 0.16666666666666669*v_conv(1,0) + 0.16666666666666669*v_conv(3,0);
    const double cRHS52 = cRHS38*(cRHS51 + 0.62200846792814624*v_conv(0,0) + 0.044658198738520456*v_conv(2,0));
    const double cRHS53 = 0.16666666666666669*v_conv(1,1) + 0.16666666666666669*v_conv(3,1);
    const double cRHS54 = cRHS44*(cRHS53 + 0.62200846792814624*v_conv(0,1) + 0.044658198738520456*v_conv(2,1));
    const double cRHS55 = cRHS29*cRHS52 + cRHS32*cRHS54;
    const double cRHS56 = cRHS38*(cRHS51 + 0.044658198738520456*v_conv(0,0) + 0.62200846792814624*v_conv(2,0));
    const double cRHS57 = cRHS44*(cRHS53 + 0.044658198738520456*v_conv(0,1) + 0.62200846792814624*v_conv(2,1));
    const double cRHS58 = cRHS19*cRHS56 + cRHS24*cRHS57;
    const double cRHS59 = cRHS38*cRHS44*mu;
    const double cRHS60 = 16.0*cRHS59;
    const double cRHS61 = 0.26794919243112275*v_conv(0,0) + 0.26794919243112275*v_conv(2,0);
    const double cRHS62 = 0.26794919243112275*v_conv(0,1) + 0.26794919243112275*v_conv(2,1);
    const double cRHS63 = 2.4880338717125849*rho/(sqrt(a)*sqrt(b));
    const double cRHS64 = 1.0/(cRHS60 + cRHS63*sqrt(pow(cRHS61 + v_conv(1,0) + 0.071796769724490839*v_conv(3,0), 2) + pow(cRHS62 + v_conv(1,1) + 0.071796769724490839*v_conv(3,1), 2)));
    const double cRHS65 = 0.16666666666666669*acc(0,0) + 0.16666666666666669*acc(2,0);
    const double cRHS66 = rho*(0.62200846792814624*acc(1,0) + 0.044658198738520456*acc(3,0) + cRHS65);
    const double cRHS67 = 0.16666666666666669*f(0,0) + 0.16666666666666669*f(2,0);
    const double cRHS68 = rho*(cRHS67 + 0.62200846792814624*f(1,0) + 0.044658198738520456*f(3,0));
    const double cRHS69 = 1.0*cRHS59;
    const double cRHS70 = cRHS69*v(1,1);
    const double cRHS71 = cRHS69*v(3,1);
    const double cRHS72 = cRHS69*v(0,1);
    const double cRHS73 = cRHS69*v(2,1);
    const double cRHS74 = cRHS70 + cRHS71 - cRHS72 - cRHS73;
    const double cRHS75 = cRHS50*rho + cRHS66 - cRHS68 + cRHS74;
    const double cRHS76 = -cRHS75;
    const double cRHS77 = cRHS64*cRHS76;
    const double cRHS78 = 0.21132486540518713*cRHS38;
    const double cRHS79 = 0.78867513459481287*cRHS38;
    const double cRHS80 = -0.78867513459481287*cRHS38*v_conv(1,0) - 0.21132486540518713*cRHS38*v_conv(2,0) + cRHS78*v_conv(3,0) + cRHS79*v_conv(0,0);
    const double cRHS81 = 0.21132486540518713*cRHS44;
    const double cRHS82 = 0.78867513459481287*cRHS44;
    const double cRHS83 = -0.78867513459481287*cRHS44*v_conv(2,1) - 0.21132486540518713*cRHS44*v_conv(3,1) + cRHS81*v_conv(0,1) + cRHS82*v_conv(1,1);
    const double cRHS84 = cRHS80 + cRHS83;
    const double cRHS85 = -cRHS84;
    const double cRHS86 = 0.16666666666666669*rho;
    const double cRHS87 = cRHS85*cRHS86;
    const double cRHS88 = 1.0/(cRHS60 + cRHS63*sqrt(pow(cRHS61 + 0.071796769724490839*v_conv(1,0) + v_conv(3,0), 2) + pow(cRHS62 + 0.071796769724490839*v_conv(1,1) + v_conv(3,1), 2)));
    const double cRHS89 = rho*(0.044658198738520456*acc(1,0) + 0.62200846792814624*acc(3,0) + cRHS65);
    const double cRHS90 = rho*(cRHS67 + 0.044658198738520456*f(1,0) + 0.62200846792814624*f(3,0));
    const double cRHS91 = cRHS47*rho + cRHS74 + cRHS89 - cRHS90;
    const double cRHS92 = -cRHS91;
    const double cRHS93 = cRHS88*cRHS92;
    const double cRHS94 = -0.21132486540518713*cRHS44*v_conv(2,1) - 0.78867513459481287*cRHS44*v_conv(3,1) + cRHS81*v_conv(1,1) + cRHS82*v_conv(0,1);
    const double cRHS95 = -0.21132486540518713*cRHS38*v_conv(1,0) - 0.78867513459481287*cRHS38*v_conv(2,0) + cRHS78*v_conv(0,0) + cRHS79*v_conv(3,0);
    const double cRHS96 = cRHS94 + cRHS95;
    const double cRHS97 = -cRHS96;
    const double cRHS98 = cRHS86*cRHS97;
    const double cRHS99 = 0.26794919243112275*v_conv(1,0) + 0.26794919243112275*v_conv(3,0);
    const double cRHS100 = 0.26794919243112275*v_conv(1,1) + 0.26794919243112275*v_conv(3,1);
    const double cRHS101 = 1.0/(cRHS60 + cRHS63*sqrt(pow(cRHS100 + v_conv(0,1) + 0.071796769724490839*v_conv(2,1), 2) + pow(cRHS99 + v_conv(0,0) + 0.071796769724490839*v_conv(2,0), 2)));
    const double cRHS102 = 0.16666666666666669*acc(1,0) + 0.16666666666666669*acc(3,0);
    const double cRHS103 = rho*(0.62200846792814624*acc(0,0) + 0.044658198738520456*acc(2,0) + cRHS102);
    const double cRHS104 = cRHS103 - cRHS36 + cRHS55*rho + cRHS74;
    const double cRHS105 = -cRHS104;
    const double cRHS106 = cRHS101*cRHS105;
    const double cRHS107 = cRHS80 + cRHS94;
    const double cRHS108 = -cRHS107;
    const double cRHS109 = 0.62200846792814624*rho;
    const double cRHS110 = cRHS108*cRHS109;
    const double cRHS111 = 1.0/(cRHS60 + cRHS63*sqrt(pow(cRHS100 + 0.071796769724490839*v_conv(0,1) + v_conv(2,1), 2) + pow(cRHS99 + 0.071796769724490839*v_conv(0,0) + v_conv(2,0), 2)));
    const double cRHS112 = rho*(0.044658198738520456*acc(0,0) + 0.62200846792814624*acc(2,0) + cRHS102);
    const double cRHS113 = cRHS112 - cRHS37 + cRHS58*rho + cRHS74;
    const double cRHS114 = -cRHS113;
    const double cRHS115 = cRHS111*cRHS114;
    const double cRHS116 = cRHS83 + cRHS95;
    const double cRHS117 = -cRHS116;
    const double cRHS118 = 0.044658198738520456*rho;
    const double cRHS119 = cRHS117*cRHS118;
    const double cRHS120 = 0.13144585576580217*v_conv(0,0);
    const double cRHS121 = 0.13144585576580217*v_conv(2,0);
    const double cRHS122 = cRHS120 + cRHS121;
    const double cRHS123 = 0.13144585576580217*v_conv(1,1);
    const double cRHS124 = 0.035220810900864527*v_conv(0,1) + 0.035220810900864527*v_conv(2,1);
    const double cRHS125 = cRHS38*(cRHS122 + 0.49056261216234409*v_conv(1,0) + 0.03522081090086452*v_conv(3,0)) + cRHS44*(cRHS123 + cRHS124 + 0.0094373878376559327*v_conv(3,1));
    const double cRHS126 = 0.13144585576580217*v_conv(3,0);
    const double cRHS127 = 0.035220810900864527*v_conv(0,0) + 0.035220810900864527*v_conv(2,0);
    const double cRHS128 = 0.13144585576580217*v_conv(0,1);
    const double cRHS129 = 0.13144585576580217*v_conv(2,1);
    const double cRHS130 = cRHS128 + cRHS129;
    const double cRHS131 = cRHS38*(cRHS126 + cRHS127 + 0.0094373878376559327*v_conv(1,0)) + cRHS44*(cRHS130 + 0.03522081090086452*v_conv(1,1) + 0.49056261216234409*v_conv(3,1));
    const double cRHS132 = cRHS56 + cRHS57;
    const double cRHS133 = cRHS52 + cRHS54;
    const double cRHS134 = 0.16666666666666669*cRHS68 + 0.16666666666666669*cRHS90;
    const double cRHS135 = 0.25*a*b;
    const double cRHS136 = 0.62200846792814624*v(0,1);
    const double cRHS137 = 0.62200846792814624*v(1,1);
    const double cRHS138 = 0.16666666666666669*v(3,1);
    const double cRHS139 = -0.16666666666666669*v(2,1);
    const double cRHS140 = cRHS138 + cRHS139;
    const double cRHS141 = -cRHS136 + cRHS137 - cRHS140;
    const double cRHS142 = 0.044658198738520456*v(0,1);
    const double cRHS143 = 0.16666666666666669*v(1,1);
    const double cRHS144 = cRHS139 + cRHS143;
    const double cRHS145 = cRHS142 + cRHS144 - 0.044658198738520456*v(3,1);
    const double cRHS146 = 0.044658198738520456*v(1,1);
    const double cRHS147 = -cRHS140 - cRHS142 + cRHS146;
    const double cRHS148 = cRHS136 + cRHS144 - 0.62200846792814624*v(3,1);
    const double cRHS149 = 0.78867513459481287*v(3,1);
    const double cRHS150 = 0.21132486540518713*v(1,1);
    const double cRHS151 = 0.21132486540518713*v(0,1) - 0.78867513459481287*v(2,1);
    const double cRHS152 = cRHS149 - cRHS150 + cRHS151;
    const double cRHS153 = -cRHS152;
    const double cRHS154 = 0.78867513459481287*v(1,1);
    const double cRHS155 = 0.21132486540518713*v(3,1);
    const double cRHS156 = cRHS151 + cRHS154 - cRHS155;
    const double cRHS157 = -cRHS156;
    const double cRHS158 = cRHS157*cRHS6;
    const double cRHS159 = 0.78867513459481287*v(0,1) - 0.21132486540518713*v(2,1);
    const double cRHS160 = -cRHS154 + cRHS155 + cRHS159;
    const double cRHS161 = -cRHS160;
    const double cRHS162 = -cRHS149 + cRHS150 + cRHS159;
    const double cRHS163 = -cRHS162;
    const double cRHS164 = cRHS163*cRHS6;
    const double cRHS165 = 0.16666666666666669*f(1,1) + 0.16666666666666669*f(3,1);
    const double cRHS166 = cRHS165 + 0.62200846792814624*f(0,1) + 0.044658198738520456*f(2,1);
    const double cRHS167 = cRHS166*rho;
    const double cRHS168 = cRHS165 + 0.044658198738520456*f(0,1) + 0.62200846792814624*f(2,1);
    const double cRHS169 = cRHS168*rho;
    const double cRHS170 = cRHS39*cRHS44;
    const double cRHS171 = -cRHS170;
    const double cRHS172 = cRHS153*cRHS43 + cRHS163*cRHS46;
    const double cRHS173 = cRHS157*cRHS49 + cRHS161*cRHS48;
    const double cRHS174 = cRHS161*cRHS52 + cRHS163*cRHS54;
    const double cRHS175 = cRHS153*cRHS56 + cRHS157*cRHS57;
    const double cRHS176 = 0.16666666666666669*acc(0,1) + 0.16666666666666669*acc(2,1);
    const double cRHS177 = 0.16666666666666669*f(0,1) + 0.16666666666666669*f(2,1);
    const double cRHS178 = cRHS177 + 0.62200846792814624*f(1,1) + 0.044658198738520456*f(3,1);
    const double cRHS179 = -1.0*cRHS38*cRHS44*mu*v(0,0) - 1.0*cRHS38*cRHS44*mu*v(2,0) + cRHS69*v(1,0) + cRHS69*v(3,0);
    const double cRHS180 = cRHS173*rho - cRHS178*rho + cRHS179 + rho*(0.62200846792814624*acc(1,1) + 0.044658198738520456*acc(3,1) + cRHS176);
    const double cRHS181 = -cRHS180;
    const double cRHS182 = cRHS181*cRHS64;
    const double cRHS183 = cRHS177 + 0.044658198738520456*f(1,1) + 0.62200846792814624*f(3,1);
    const double cRHS184 = cRHS172*rho + cRHS179 - cRHS183*rho + rho*(0.044658198738520456*acc(1,1) + 0.62200846792814624*acc(3,1) + cRHS176);
    const double cRHS185 = -cRHS184;
    const double cRHS186 = cRHS185*cRHS88;
    const double cRHS187 = 0.16666666666666669*acc(1,1) + 0.16666666666666669*acc(3,1);
    const double cRHS188 = -cRHS166*rho + cRHS174*rho + cRHS179 + rho*(0.62200846792814624*acc(0,1) + 0.044658198738520456*acc(2,1) + cRHS187);
    const double cRHS189 = -cRHS188;
    const double cRHS190 = cRHS101*cRHS189;
    const double cRHS191 = -cRHS168*rho + cRHS175*rho + cRHS179 + rho*(0.044658198738520456*acc(0,1) + 0.62200846792814624*acc(2,1) + cRHS187);
    const double cRHS192 = -cRHS191;
    const double cRHS193 = cRHS111*cRHS192;
    const double cRHS194 = cRHS178*rho;
    const double cRHS195 = cRHS183*rho;
    const double cRHS196 = 0.16666666666666669*cRHS194 + 0.16666666666666669*cRHS195;
    const double cRHS197 = -0.044658198738520456*v(2,0);
    const double cRHS198 = 0.16666666666666669*v(0,0);
    const double cRHS199 = cRHS198 - cRHS2;
    const double cRHS200 = cRHS6*(cRHS197 + cRHS199 + 0.044658198738520456*v(1,0));
    const double cRHS201 = -0.62200846792814624*v(2,0);
    const double cRHS202 = cRHS6*(cRHS199 + cRHS201 + 0.62200846792814624*v(1,0));
    const double cRHS203 = cRHS31*cRHS6;
    const double cRHS204 = rho*(cRHS18*cRHS56 + cRHS23*cRHS57);
    const double cRHS205 = rho*(cRHS28*cRHS52 + cRHS31*cRHS54);
    const double cRHS206 = cRHS23*cRHS6;
    const double cRHS207 = rho*(cRHS23*cRHS49 + cRHS28*cRHS48);
    const double cRHS208 = rho*(cRHS18*cRHS43 + cRHS31*cRHS46);
    const double cRHS209 = 0.035220810900864527*v_conv(1,1) + 0.035220810900864527*v_conv(3,1);
    const double cRHS210 = 0.13144585576580217*v_conv(1,0);
    const double cRHS211 = cRHS126 + cRHS210;
    const double cRHS212 = -cRHS38*(cRHS211 + 0.49056261216234409*v_conv(0,0) + 0.03522081090086452*v_conv(2,0)) + cRHS44*(cRHS128 + cRHS209 + 0.0094373878376559327*v_conv(2,1));
    const double cRHS213 = 1.0*rho;
    const double cRHS214 = -cRHS70 - cRHS71 + cRHS72 + cRHS73;
    const double cRHS215 = cRHS101*(cRHS103 - cRHS205 - cRHS214 - cRHS36);
    const double cRHS216 = 0.13144585576580217*v_conv(3,1);
    const double cRHS217 = cRHS123 + cRHS216;
    const double cRHS218 = 0.035220810900864527*v_conv(1,0) + 0.035220810900864527*v_conv(3,0);
    const double cRHS219 = -cRHS38*(cRHS121 + cRHS218 + 0.0094373878376559327*v_conv(0,0)) + cRHS44*(cRHS217 + 0.03522081090086452*v_conv(0,1) + 0.49056261216234409*v_conv(2,1));
    const double cRHS220 = cRHS111*(cRHS112 - cRHS204 - cRHS214 - cRHS37);
    const double cRHS221 = -cRHS43 + cRHS46;
    const double cRHS222 = 0.21132486540518713*rho;
    const double cRHS223 = cRHS88*(-cRHS208 - cRHS214 + cRHS89 - cRHS90);
    const double cRHS224 = cRHS107*cRHS86;
    const double cRHS225 = cRHS116*cRHS86;
    const double cRHS226 = -cRHS48 + cRHS49;
    const double cRHS227 = 0.78867513459481287*rho;
    const double cRHS228 = cRHS64*(-cRHS207 - cRHS214 + cRHS66 - cRHS68);
    const double cRHS229 = 0.16666666666666669*cRHS36 + 0.16666666666666669*cRHS37;
    const double cRHS230 = -0.044658198738520456*v(2,1);
    const double cRHS231 = 0.16666666666666669*v(0,1);
    const double cRHS232 = -cRHS138 + cRHS231;
    const double cRHS233 = cRHS146 + cRHS230 + cRHS232;
    const double cRHS234 = -0.62200846792814624*v(2,1);
    const double cRHS235 = cRHS137 + cRHS232 + cRHS234;
    const double cRHS236 = cRHS108*cRHS86;
    const double cRHS237 = cRHS117*cRHS86;
    const double cRHS238 = 0.16666666666666669*cRHS167 + 0.16666666666666669*cRHS169;
    const double cRHS239 = cRHS198 - cRHS9;
    const double cRHS240 = cRHS13 + cRHS201 + cRHS239;
    const double cRHS241 = cRHS197 + cRHS239 + cRHS8;
    const double cRHS242 = cRHS84*cRHS86;
    const double cRHS243 = cRHS86*cRHS96;
    const double cRHS244 = cRHS109*cRHS116;
    const double cRHS245 = cRHS107*cRHS118;
    const double cRHS246 = cRHS38*(cRHS127 + cRHS210 + 0.0094373878376559327*v_conv(3,0)) + cRHS44*(cRHS130 + 0.49056261216234409*v_conv(1,1) + 0.03522081090086452*v_conv(3,1));
    const double cRHS247 = cRHS38*(cRHS122 + 0.03522081090086452*v_conv(1,0) + 0.49056261216234409*v_conv(3,0)) + cRHS44*(cRHS124 + cRHS216 + 0.0094373878376559327*v_conv(1,1));
    const double cRHS248 = -cRHS143 + cRHS231;
    const double cRHS249 = cRHS234 + cRHS248 + 0.62200846792814624*v(3,1);
    const double cRHS250 = cRHS230 + cRHS248 + 0.044658198738520456*v(3,1);
    const double cRHS251 = cRHS162*cRHS6;
    const double cRHS252 = cRHS152*cRHS43 + cRHS162*cRHS46;
    const double cRHS253 = cRHS156*cRHS49 + cRHS160*cRHS48;
    const double cRHS254 = cRHS156*cRHS6;
    const double cRHS255 = cRHS152*cRHS56 + cRHS156*cRHS57;
    const double cRHS256 = cRHS160*cRHS52 + cRHS162*cRHS54;
    const double cRHS257 = cRHS180*cRHS64;
    const double cRHS258 = cRHS184*cRHS88;
    const double cRHS259 = cRHS111*cRHS191;
    const double cRHS260 = cRHS101*cRHS188;
    const double cRHS261 = -cRHS38*(cRHS120 + cRHS218 + 0.0094373878376559327*v_conv(2,0)) + cRHS44*(cRHS217 + 0.49056261216234409*v_conv(0,1) + 0.03522081090086452*v_conv(2,1));
    const double cRHS262 = -cRHS38*(cRHS211 + 0.03522081090086452*v_conv(0,0) + 0.49056261216234409*v_conv(2,0)) + cRHS44*(cRHS129 + cRHS209 + 0.0094373878376559327*v_conv(0,1));
    RHS[0] = -cRHS135*(0.78867513459481287*cRHS101*cRHS105*cRHS133*rho - cRHS106*cRHS110 + 0.21132486540518713*cRHS111*cRHS114*cRHS132*rho - cRHS115*cRHS119 + 1.0*cRHS125*cRHS64*cRHS76*rho + 1.0*cRHS131*cRHS88*cRHS92*rho - cRHS134 - cRHS26*(cRHS20 + cRHS25) - cRHS34*(cRHS30 + cRHS33) - 0.62200846792814624*cRHS36 - 0.044658198738520456*cRHS37 - cRHS41 + 0.16666666666666669*cRHS47*rho + 0.16666666666666669*cRHS50*rho + 0.62200846792814624*cRHS55*rho + 0.044658198738520456*cRHS58*rho - cRHS77*cRHS87 - cRHS93*cRHS98 - mu*(-cRHS0*cRHS12 + cRHS14) - mu*(-cRHS0*cRHS5 + cRHS11));
    RHS[1] = -cRHS135*(0.78867513459481287*cRHS101*cRHS133*cRHS189*rho - cRHS110*cRHS190 + 0.21132486540518713*cRHS111*cRHS132*cRHS192*rho - cRHS119*cRHS193 + 1.0*cRHS125*cRHS181*cRHS64*rho + 1.0*cRHS131*cRHS185*cRHS88*rho - 0.62200846792814624*cRHS167 - 0.044658198738520456*cRHS169 - cRHS171 + 0.16666666666666669*cRHS172*rho + 0.16666666666666669*cRHS173*rho + 0.62200846792814624*cRHS174*rho + 0.044658198738520456*cRHS175*rho - cRHS182*cRHS87 - cRHS186*cRHS98 - cRHS196 - cRHS26*(cRHS0*cRHS153 + cRHS158) - cRHS34*(cRHS0*cRHS161 + cRHS164) - mu*(cRHS0*cRHS141 - cRHS145*cRHS6) - mu*(cRHS0*cRHS147 - cRHS148*cRHS6));
    RHS[2] = cRHS135*(cRHS109*cRHS228*cRHS84 + cRHS118*cRHS223*cRHS96 + 0.16666666666666669*cRHS204 + 0.16666666666666669*cRHS205 + 0.62200846792814624*cRHS207 + 0.044658198738520456*cRHS208 + cRHS212*cRHS213*cRHS215 + cRHS213*cRHS219*cRHS220 + cRHS215*cRHS224 + cRHS220*cRHS225 + cRHS221*cRHS222*cRHS223 + cRHS226*cRHS227*cRHS228 + cRHS229 + cRHS26*(cRHS0*cRHS18 - cRHS203) + cRHS34*(cRHS0*cRHS28 - cRHS206) + cRHS40 + 0.62200846792814624*cRHS68 + 0.044658198738520456*cRHS90 + mu*(cRHS0*cRHS12 - cRHS202) + mu*(cRHS0*cRHS5 - cRHS200));
    RHS[3] = -cRHS135*(-cRHS109*cRHS182*cRHS85 - cRHS118*cRHS186*cRHS97 - cRHS171 + 0.044658198738520456*cRHS172*rho + 0.62200846792814624*cRHS173*rho + 0.16666666666666669*cRHS174*rho + 0.16666666666666669*cRHS175*rho + cRHS182*cRHS226*cRHS227 + cRHS186*cRHS221*cRHS222 + cRHS190*cRHS212*cRHS213 - cRHS190*cRHS236 + cRHS193*cRHS213*cRHS219 - cRHS193*cRHS237 - 0.62200846792814624*cRHS194 - 0.044658198738520456*cRHS195 - cRHS238 + mu*(cRHS0*cRHS141 + cRHS233*cRHS6) + mu*(cRHS0*cRHS147 + cRHS235*cRHS6) + 0.21132486540518713*mu*(cRHS0*cRHS153 - cRHS164) + 0.78867513459481287*mu*(cRHS0*cRHS161 - cRHS158));
    RHS[4] = -cRHS135*(0.21132486540518713*cRHS101*cRHS104*cRHS133*rho - cRHS101*cRHS104*cRHS245 + 0.78867513459481287*cRHS111*cRHS113*cRHS132*rho - cRHS111*cRHS113*cRHS244 - cRHS134 - 0.62200846792814624*cRHS204 - 0.044658198738520456*cRHS205 - 0.16666666666666669*cRHS207 - 0.16666666666666669*cRHS208 - cRHS242*cRHS64*cRHS75 - cRHS243*cRHS88*cRHS91 + 1.0*cRHS246*cRHS64*cRHS75*rho + 1.0*cRHS247*cRHS88*cRHS91*rho - cRHS26*(cRHS0*cRHS28 + cRHS203) - cRHS34*(cRHS0*cRHS18 + cRHS206) - 0.044658198738520456*cRHS36 - 0.62200846792814624*cRHS37 - cRHS40 - mu*(cRHS0*cRHS240 + cRHS200) - mu*(cRHS0*cRHS241 + cRHS202));
    RHS[5] = -cRHS135*(0.21132486540518713*cRHS101*cRHS133*cRHS188*rho - cRHS109*cRHS255 + 0.78867513459481287*cRHS111*cRHS132*cRHS191*rho - cRHS118*cRHS256 - 0.044658198738520456*cRHS167 - 0.62200846792814624*cRHS169 - cRHS170 + 1.0*cRHS180*cRHS246*cRHS64*rho + 1.0*cRHS184*cRHS247*cRHS88*rho - cRHS196 - cRHS242*cRHS257 - cRHS243*cRHS258 - cRHS244*cRHS259 - cRHS245*cRHS260 - cRHS252*cRHS86 - cRHS253*cRHS86 - cRHS26*(cRHS0*cRHS160 + cRHS251) - cRHS34*(cRHS0*cRHS152 + cRHS254) - mu*(cRHS0*cRHS249 + cRHS233*cRHS6) - mu*(cRHS0*cRHS250 + cRHS235*cRHS6));
    RHS[6] = -cRHS135*(-cRHS106*cRHS213*cRHS261 - cRHS106*cRHS236 - cRHS109*cRHS93*cRHS97 - cRHS115*cRHS213*cRHS262 - cRHS115*cRHS237 - cRHS118*cRHS77*cRHS85 - cRHS221*cRHS227*cRHS93 - cRHS222*cRHS226*cRHS77 - cRHS229 - cRHS41 + 0.62200846792814624*cRHS47*rho + 0.044658198738520456*cRHS50*rho + 0.16666666666666669*cRHS55*rho + 0.16666666666666669*cRHS58*rho - 0.044658198738520456*cRHS68 - 0.62200846792814624*cRHS90 + 0.78867513459481287*mu*(-cRHS20 + cRHS33) + 0.21132486540518713*mu*(cRHS25 - cRHS30) + mu*(cRHS0*cRHS240 + cRHS11) + mu*(cRHS0*cRHS241 + cRHS14));
    RHS[7] = -cRHS135*(1.0*cRHS101*cRHS188*cRHS261*rho - cRHS109*cRHS252 - cRHS109*cRHS258*cRHS96 + 1.0*cRHS111*cRHS191*cRHS262*rho - cRHS118*cRHS253 - cRHS118*cRHS257*cRHS84 - cRHS170 + 0.21132486540518713*cRHS180*cRHS226*cRHS64*rho + 0.78867513459481287*cRHS184*cRHS221*cRHS88*rho - 0.044658198738520456*cRHS194 - 0.62200846792814624*cRHS195 - cRHS224*cRHS260 - cRHS225*cRHS259 - cRHS238 - cRHS255*cRHS86 - cRHS256*cRHS86 + 0.78867513459481287*mu*(cRHS0*cRHS152 - cRHS251) + 0.21132486540518713*mu*(cRHS0*cRHS160 - cRHS254) + mu*(cRHS0*cRHS249 - cRHS145*cRHS6) + mu*(cRHS0*cRHS250 - cRHS148*cRHS6));

}

void IncompressibleNavierStokesQ1P0StructuredElement::GetCellGradientOperator(
    const double a,
    const double b,
    Eigen::Array<double, 4, 2>& G)
{

    const double cG0 = 0.5*b;
    const double cG1 = -cG0;
    const double cG2 = 0.5*a;
    const double cG3 = -cG2;
    G(0,0) = cG1;
    G(0,1) = cG3;
    G(1,0) = cG0;
    G(1,1) = cG3;
    G(2,0) = cG0;
    G(2,1) = cG2;
    G(3,0) = cG1;
    G(3,1) = cG2;

}

void IncompressibleNavierStokesQ1P0StructuredElement::CalculateRightHandSide(
    const double a,
    const double b,
    const double c,
    const double mu,
    const double rho,
    const Eigen::Array<double, 8, 3>& v,
    const double p,
    const Eigen::Array<double, 8, 3>& f,
    const Eigen::Array<double, 8, 3>& acc,
    Eigen::Array<double, 24, 1>& RHS)
{

//substitute_rhs_3d
}

void IncompressibleNavierStokesQ1P0StructuredElement::GetCellGradientOperator(
    const double a,
    const double b,
    const double c,
    Eigen::Array<double, 8, 3>& G)
{

//substitute_G_3d
}
