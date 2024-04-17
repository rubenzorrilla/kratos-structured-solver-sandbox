#include "incompressible_navier_stokes_q1_p0_structured_element.hpp"

void IncompressibleNavierStokesQ1P0StructuredElement::CalculateRightHandSide(
    const double a,
    const double b,
    const double mu,
    const double rho,
    const QuadVectorDataView& v,
    const double p,
    const QuadVectorDataView& f,
    const QuadVectorDataView& acc,
    std::array<double, 8>& RHS)
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
    const double cRHS17 = 0.21132486540518713*v(0,0);
    const double cRHS18 = cRHS17 - 0.78867513459481287*v(2,0);
    const double cRHS19 = cRHS15 - cRHS16 + cRHS18;
    const double cRHS20 = -cRHS19;
    const double cRHS21 = cRHS0*cRHS20;
    const double cRHS22 = 0.78867513459481287*v(1,0);
    const double cRHS23 = 0.21132486540518713*v(3,0);
    const double cRHS24 = cRHS18 + cRHS22 - cRHS23;
    const double cRHS25 = -cRHS24;
    const double cRHS26 = cRHS25*cRHS6;
    const double cRHS27 = 0.21132486540518713*mu;
    const double cRHS28 = 0.78867513459481287*v(0,0);
    const double cRHS29 = cRHS28 - 0.21132486540518713*v(2,0);
    const double cRHS30 = -cRHS22 + cRHS23 + cRHS29;
    const double cRHS31 = -cRHS30;
    const double cRHS32 = cRHS0*cRHS31;
    const double cRHS33 = -cRHS15 + cRHS16 + cRHS29;
    const double cRHS34 = -cRHS33;
    const double cRHS35 = cRHS34*cRHS6;
    const double cRHS36 = 0.78867513459481287*mu;
    const double cRHS37 = 0.16666666666666669*f(1,0) + 0.16666666666666669*f(3,0);
    const double cRHS38 = rho*(cRHS37 + 0.62200846792814624*f(0,0) + 0.044658198738520456*f(2,0));
    const double cRHS39 = rho*(cRHS37 + 0.044658198738520456*f(0,0) + 0.62200846792814624*f(2,0));
    const double cRHS40 = 1.0/a;
    const double cRHS41 = 2.0*p;
    const double cRHS42 = cRHS40*cRHS41;
    const double cRHS43 = -cRHS42;
    const double cRHS44 = 0.044658198738520456*v(1,0);
    const double cRHS45 = 0.16666666666666669*v(0,0);
    const double cRHS46 = cRHS45 + 0.16666666666666669*v(2,0);
    const double cRHS47 = cRHS40*(cRHS13 + cRHS44 + cRHS46);
    const double cRHS48 = 1.0/b;
    const double cRHS49 = 0.62200846792814624*v(3,1);
    const double cRHS50 = 0.044658198738520456*v(1,1);
    const double cRHS51 = 0.16666666666666669*v(0,1);
    const double cRHS52 = cRHS51 + 0.16666666666666669*v(2,1);
    const double cRHS53 = cRHS48*(cRHS49 + cRHS50 + cRHS52);
    const double cRHS54 = cRHS20*cRHS47 + cRHS34*cRHS53;
    const double cRHS55 = 0.62200846792814624*v(1,0);
    const double cRHS56 = cRHS40*(cRHS46 + cRHS55 + cRHS8);
    const double cRHS57 = 0.62200846792814624*v(1,1);
    const double cRHS58 = 0.044658198738520456*v(3,1);
    const double cRHS59 = cRHS48*(cRHS52 + cRHS57 + cRHS58);
    const double cRHS60 = cRHS25*cRHS59 + cRHS31*cRHS56;
    const double cRHS61 = 0.044658198738520456*v(2,0);
    const double cRHS62 = cRHS2 + cRHS9;
    const double cRHS63 = cRHS40*(cRHS1 + cRHS61 + cRHS62);
    const double cRHS64 = 0.62200846792814624*v(0,1);
    const double cRHS65 = 0.044658198738520456*v(2,1);
    const double cRHS66 = 0.16666666666666669*v(1,1);
    const double cRHS67 = 0.16666666666666669*v(3,1);
    const double cRHS68 = cRHS66 + cRHS67;
    const double cRHS69 = cRHS48*(cRHS64 + cRHS65 + cRHS68);
    const double cRHS70 = cRHS31*cRHS63 + cRHS34*cRHS69;
    const double cRHS71 = 0.62200846792814624*v(2,0);
    const double cRHS72 = cRHS40*(cRHS62 + cRHS7 + cRHS71);
    const double cRHS73 = 0.62200846792814624*v(2,1);
    const double cRHS74 = 0.044658198738520456*v(0,1);
    const double cRHS75 = cRHS48*(cRHS68 + cRHS73 + cRHS74);
    const double cRHS76 = cRHS20*cRHS72 + cRHS25*cRHS75;
    const double cRHS77 = cRHS40*cRHS48*mu;
    const double cRHS78 = 16.0*cRHS77;
    const double cRHS79 = 0.26794919243112275*v(0,0) + 0.26794919243112275*v(2,0);
    const double cRHS80 = 0.26794919243112275*v(0,1) + 0.26794919243112275*v(2,1);
    const double cRHS81 = 2.4880338717125849*rho/(sqrt(a)*sqrt(b));
    const double cRHS82 = 1.0/(cRHS78 + cRHS81*sqrt(pow(cRHS79 + v(1,0) + 0.071796769724490839*v(3,0), 2) + pow(cRHS80 + v(1,1) + 0.071796769724490839*v(3,1), 2)));
    const double cRHS83 = 0.16666666666666669*acc(0,0) + 0.16666666666666669*acc(2,0);
    const double cRHS84 = rho*(0.62200846792814624*acc(1,0) + 0.044658198738520456*acc(3,0) + cRHS83);
    const double cRHS85 = 0.16666666666666669*f(0,0) + 0.16666666666666669*f(2,0);
    const double cRHS86 = rho*(cRHS85 + 0.62200846792814624*f(1,0) + 0.044658198738520456*f(3,0));
    const double cRHS87 = 1.0*cRHS77;
    const double cRHS88 = cRHS87*v(1,1);
    const double cRHS89 = cRHS87*v(3,1);
    const double cRHS90 = cRHS87*v(0,1);
    const double cRHS91 = cRHS87*v(2,1);
    const double cRHS92 = cRHS88 + cRHS89 - cRHS90 - cRHS91;
    const double cRHS93 = cRHS60*rho + cRHS84 - cRHS86 + cRHS92;
    const double cRHS94 = -cRHS93;
    const double cRHS95 = cRHS82*cRHS94;
    const double cRHS96 = cRHS23*cRHS40 + cRHS28*cRHS40 - 0.78867513459481287*cRHS40*v(1,0) - 0.21132486540518713*cRHS40*v(2,0);
    const double cRHS97 = 0.21132486540518713*v(0,1);
    const double cRHS98 = 0.78867513459481287*v(1,1);
    const double cRHS99 = cRHS48*cRHS97 + cRHS48*cRHS98 - 0.78867513459481287*cRHS48*v(2,1) - 0.21132486540518713*cRHS48*v(3,1);
    const double cRHS100 = cRHS96 + cRHS99;
    const double cRHS101 = -cRHS100;
    const double cRHS102 = 0.16666666666666669*rho;
    const double cRHS103 = cRHS101*cRHS102;
    const double cRHS104 = 1.0/(cRHS78 + cRHS81*sqrt(pow(cRHS79 + 0.071796769724490839*v(1,0) + v(3,0), 2) + pow(cRHS80 + 0.071796769724490839*v(1,1) + v(3,1), 2)));
    const double cRHS105 = rho*(0.044658198738520456*acc(1,0) + 0.62200846792814624*acc(3,0) + cRHS83);
    const double cRHS106 = rho*(cRHS85 + 0.044658198738520456*f(1,0) + 0.62200846792814624*f(3,0));
    const double cRHS107 = cRHS105 - cRHS106 + cRHS54*rho + cRHS92;
    const double cRHS108 = -cRHS107;
    const double cRHS109 = cRHS104*cRHS108;
    const double cRHS110 = 0.21132486540518713*v(1,1);
    const double cRHS111 = 0.78867513459481287*v(0,1);
    const double cRHS112 = cRHS110*cRHS48 + cRHS111*cRHS48 - 0.21132486540518713*cRHS48*v(2,1) - 0.78867513459481287*cRHS48*v(3,1);
    const double cRHS113 = cRHS15*cRHS40 + cRHS17*cRHS40 - 0.21132486540518713*cRHS40*v(1,0) - 0.78867513459481287*cRHS40*v(2,0);
    const double cRHS114 = cRHS112 + cRHS113;
    const double cRHS115 = -cRHS114;
    const double cRHS116 = cRHS102*cRHS115;
    const double cRHS117 = 0.26794919243112275*v(1,0) + 0.26794919243112275*v(3,0);
    const double cRHS118 = 0.26794919243112275*v(1,1) + 0.26794919243112275*v(3,1);
    const double cRHS119 = 1.0/(cRHS78 + cRHS81*sqrt(pow(cRHS117 + v(0,0) + 0.071796769724490839*v(2,0), 2) + pow(cRHS118 + v(0,1) + 0.071796769724490839*v(2,1), 2)));
    const double cRHS120 = 0.16666666666666669*acc(1,0) + 0.16666666666666669*acc(3,0);
    const double cRHS121 = rho*(0.62200846792814624*acc(0,0) + 0.044658198738520456*acc(2,0) + cRHS120);
    const double cRHS122 = cRHS121 - cRHS38 + cRHS70*rho + cRHS92;
    const double cRHS123 = -cRHS122;
    const double cRHS124 = cRHS119*cRHS123;
    const double cRHS125 = cRHS112 + cRHS96;
    const double cRHS126 = -cRHS125;
    const double cRHS127 = 0.62200846792814624*rho;
    const double cRHS128 = cRHS126*cRHS127;
    const double cRHS129 = 1.0/(cRHS78 + cRHS81*sqrt(pow(cRHS117 + 0.071796769724490839*v(0,0) + v(2,0), 2) + pow(cRHS118 + 0.071796769724490839*v(0,1) + v(2,1), 2)));
    const double cRHS130 = rho*(0.044658198738520456*acc(0,0) + 0.62200846792814624*acc(2,0) + cRHS120);
    const double cRHS131 = cRHS130 - cRHS39 + cRHS76*rho + cRHS92;
    const double cRHS132 = -cRHS131;
    const double cRHS133 = cRHS129*cRHS132;
    const double cRHS134 = cRHS113 + cRHS99;
    const double cRHS135 = -cRHS134;
    const double cRHS136 = 0.044658198738520456*rho;
    const double cRHS137 = cRHS135*cRHS136;
    const double cRHS138 = 0.13144585576580217*v(0,0);
    const double cRHS139 = 0.13144585576580217*v(2,0);
    const double cRHS140 = cRHS138 + cRHS139;
    const double cRHS141 = 0.13144585576580217*v(1,1);
    const double cRHS142 = 0.035220810900864527*v(0,1) + 0.035220810900864527*v(2,1);
    const double cRHS143 = cRHS40*(cRHS140 + 0.49056261216234409*v(1,0) + 0.03522081090086452*v(3,0)) + cRHS48*(cRHS141 + cRHS142 + 0.0094373878376559327*v(3,1));
    const double cRHS144 = 0.13144585576580217*v(3,0);
    const double cRHS145 = 0.035220810900864527*v(0,0) + 0.035220810900864527*v(2,0);
    const double cRHS146 = 0.13144585576580217*v(0,1);
    const double cRHS147 = 0.13144585576580217*v(2,1);
    const double cRHS148 = cRHS146 + cRHS147;
    const double cRHS149 = cRHS40*(cRHS144 + cRHS145 + 0.0094373878376559327*v(1,0)) + cRHS48*(cRHS148 + 0.03522081090086452*v(1,1) + 0.49056261216234409*v(3,1));
    const double cRHS150 = cRHS72 + cRHS75;
    const double cRHS151 = cRHS63 + cRHS69;
    const double cRHS152 = 0.16666666666666669*cRHS106 + 0.16666666666666669*cRHS86;
    const double cRHS153 = 0.25*a*b;
    const double cRHS154 = -0.16666666666666669*v(2,1);
    const double cRHS155 = cRHS154 + cRHS67;
    const double cRHS156 = -cRHS155 + cRHS57 - cRHS64;
    const double cRHS157 = cRHS154 + cRHS66;
    const double cRHS158 = cRHS157 + cRHS74 - 0.044658198738520456*v(3,1);
    const double cRHS159 = -cRHS155 + cRHS50 - cRHS74;
    const double cRHS160 = cRHS157 + cRHS64 - 0.62200846792814624*v(3,1);
    const double cRHS161 = 0.78867513459481287*v(3,1);
    const double cRHS162 = cRHS97 - 0.78867513459481287*v(2,1);
    const double cRHS163 = -cRHS110 + cRHS161 + cRHS162;
    const double cRHS164 = -cRHS163;
    const double cRHS165 = 0.21132486540518713*v(3,1);
    const double cRHS166 = cRHS162 - cRHS165 + cRHS98;
    const double cRHS167 = -cRHS166;
    const double cRHS168 = cRHS167*cRHS6;
    const double cRHS169 = cRHS111 - 0.21132486540518713*v(2,1);
    const double cRHS170 = cRHS165 + cRHS169 - cRHS98;
    const double cRHS171 = -cRHS170;
    const double cRHS172 = cRHS110 - cRHS161 + cRHS169;
    const double cRHS173 = -cRHS172;
    const double cRHS174 = cRHS173*cRHS6;
    const double cRHS175 = 0.16666666666666669*f(1,1) + 0.16666666666666669*f(3,1);
    const double cRHS176 = cRHS175 + 0.62200846792814624*f(0,1) + 0.044658198738520456*f(2,1);
    const double cRHS177 = cRHS176*rho;
    const double cRHS178 = cRHS175 + 0.044658198738520456*f(0,1) + 0.62200846792814624*f(2,1);
    const double cRHS179 = cRHS178*rho;
    const double cRHS180 = cRHS41*cRHS48;
    const double cRHS181 = -cRHS180;
    const double cRHS182 = cRHS164*cRHS47 + cRHS173*cRHS53;
    const double cRHS183 = cRHS167*cRHS59 + cRHS171*cRHS56;
    const double cRHS184 = cRHS171*cRHS63 + cRHS173*cRHS69;
    const double cRHS185 = cRHS164*cRHS72 + cRHS167*cRHS75;
    const double cRHS186 = 0.16666666666666669*acc(0,1) + 0.16666666666666669*acc(2,1);
    const double cRHS187 = 0.16666666666666669*f(0,1) + 0.16666666666666669*f(2,1);
    const double cRHS188 = cRHS187 + 0.62200846792814624*f(1,1) + 0.044658198738520456*f(3,1);
    const double cRHS189 = -1.0*cRHS40*cRHS48*mu*v(0,0) - 1.0*cRHS40*cRHS48*mu*v(2,0) + cRHS87*v(1,0) + cRHS87*v(3,0);
    const double cRHS190 = cRHS183*rho - cRHS188*rho + cRHS189 + rho*(0.62200846792814624*acc(1,1) + 0.044658198738520456*acc(3,1) + cRHS186);
    const double cRHS191 = -cRHS190;
    const double cRHS192 = cRHS191*cRHS82;
    const double cRHS193 = cRHS187 + 0.044658198738520456*f(1,1) + 0.62200846792814624*f(3,1);
    const double cRHS194 = cRHS182*rho + cRHS189 - cRHS193*rho + rho*(0.044658198738520456*acc(1,1) + 0.62200846792814624*acc(3,1) + cRHS186);
    const double cRHS195 = -cRHS194;
    const double cRHS196 = cRHS104*cRHS195;
    const double cRHS197 = 0.16666666666666669*acc(1,1) + 0.16666666666666669*acc(3,1);
    const double cRHS198 = -cRHS176*rho + cRHS184*rho + cRHS189 + rho*(0.62200846792814624*acc(0,1) + 0.044658198738520456*acc(2,1) + cRHS197);
    const double cRHS199 = -cRHS198;
    const double cRHS200 = cRHS119*cRHS199;
    const double cRHS201 = -cRHS178*rho + cRHS185*rho + cRHS189 + rho*(0.044658198738520456*acc(0,1) + 0.62200846792814624*acc(2,1) + cRHS197);
    const double cRHS202 = -cRHS201;
    const double cRHS203 = cRHS129*cRHS202;
    const double cRHS204 = cRHS188*rho;
    const double cRHS205 = cRHS193*rho;
    const double cRHS206 = 0.16666666666666669*cRHS204 + 0.16666666666666669*cRHS205;
    const double cRHS207 = -cRHS61;
    const double cRHS208 = -cRHS2 + cRHS45;
    const double cRHS209 = cRHS6*(cRHS207 + cRHS208 + cRHS44);
    const double cRHS210 = -cRHS71;
    const double cRHS211 = cRHS6*(cRHS208 + cRHS210 + cRHS55);
    const double cRHS212 = cRHS33*cRHS6;
    const double cRHS213 = rho*(cRHS19*cRHS72 + cRHS24*cRHS75);
    const double cRHS214 = rho*(cRHS30*cRHS63 + cRHS33*cRHS69);
    const double cRHS215 = cRHS24*cRHS6;
    const double cRHS216 = rho*(cRHS24*cRHS59 + cRHS30*cRHS56);
    const double cRHS217 = rho*(cRHS19*cRHS47 + cRHS33*cRHS53);
    const double cRHS218 = 0.035220810900864527*v(1,1) + 0.035220810900864527*v(3,1);
    const double cRHS219 = 0.13144585576580217*v(1,0);
    const double cRHS220 = cRHS144 + cRHS219;
    const double cRHS221 = -cRHS40*(cRHS220 + 0.49056261216234409*v(0,0) + 0.03522081090086452*v(2,0)) + cRHS48*(cRHS146 + cRHS218 + 0.0094373878376559327*v(2,1));
    const double cRHS222 = 1.0*rho;
    const double cRHS223 = -cRHS88 - cRHS89 + cRHS90 + cRHS91;
    const double cRHS224 = cRHS119*(cRHS121 - cRHS214 - cRHS223 - cRHS38);
    const double cRHS225 = 0.13144585576580217*v(3,1);
    const double cRHS226 = cRHS141 + cRHS225;
    const double cRHS227 = 0.035220810900864527*v(1,0) + 0.035220810900864527*v(3,0);
    const double cRHS228 = -cRHS40*(cRHS139 + cRHS227 + 0.0094373878376559327*v(0,0)) + cRHS48*(cRHS226 + 0.03522081090086452*v(0,1) + 0.49056261216234409*v(2,1));
    const double cRHS229 = cRHS129*(cRHS130 - cRHS213 - cRHS223 - cRHS39);
    const double cRHS230 = -cRHS47 + cRHS53;
    const double cRHS231 = 0.21132486540518713*rho;
    const double cRHS232 = cRHS104*(cRHS105 - cRHS106 - cRHS217 - cRHS223);
    const double cRHS233 = cRHS102*cRHS125;
    const double cRHS234 = cRHS102*cRHS134;
    const double cRHS235 = -cRHS56 + cRHS59;
    const double cRHS236 = 0.78867513459481287*rho;
    const double cRHS237 = cRHS82*(-cRHS216 - cRHS223 + cRHS84 - cRHS86);
    const double cRHS238 = 0.16666666666666669*cRHS38 + 0.16666666666666669*cRHS39;
    const double cRHS239 = -cRHS65;
    const double cRHS240 = cRHS51 - cRHS67;
    const double cRHS241 = cRHS239 + cRHS240 + cRHS50;
    const double cRHS242 = -cRHS73;
    const double cRHS243 = cRHS240 + cRHS242 + cRHS57;
    const double cRHS244 = cRHS102*cRHS126;
    const double cRHS245 = cRHS102*cRHS135;
    const double cRHS246 = 0.16666666666666669*cRHS177 + 0.16666666666666669*cRHS179;
    const double cRHS247 = cRHS45 - cRHS9;
    const double cRHS248 = cRHS13 + cRHS210 + cRHS247;
    const double cRHS249 = cRHS207 + cRHS247 + cRHS8;
    const double cRHS250 = cRHS100*cRHS102;
    const double cRHS251 = cRHS102*cRHS114;
    const double cRHS252 = cRHS127*cRHS134;
    const double cRHS253 = cRHS125*cRHS136;
    const double cRHS254 = cRHS40*(cRHS145 + cRHS219 + 0.0094373878376559327*v(3,0)) + cRHS48*(cRHS148 + 0.49056261216234409*v(1,1) + 0.03522081090086452*v(3,1));
    const double cRHS255 = cRHS40*(cRHS140 + 0.03522081090086452*v(1,0) + 0.49056261216234409*v(3,0)) + cRHS48*(cRHS142 + cRHS225 + 0.0094373878376559327*v(1,1));
    const double cRHS256 = cRHS51 - cRHS66;
    const double cRHS257 = cRHS242 + cRHS256 + cRHS49;
    const double cRHS258 = cRHS239 + cRHS256 + cRHS58;
    const double cRHS259 = cRHS172*cRHS6;
    const double cRHS260 = cRHS163*cRHS47 + cRHS172*cRHS53;
    const double cRHS261 = cRHS166*cRHS59 + cRHS170*cRHS56;
    const double cRHS262 = cRHS166*cRHS6;
    const double cRHS263 = cRHS163*cRHS72 + cRHS166*cRHS75;
    const double cRHS264 = cRHS170*cRHS63 + cRHS172*cRHS69;
    const double cRHS265 = cRHS190*cRHS82;
    const double cRHS266 = cRHS104*cRHS194;
    const double cRHS267 = cRHS129*cRHS201;
    const double cRHS268 = cRHS119*cRHS198;
    const double cRHS269 = -cRHS40*(cRHS138 + cRHS227 + 0.0094373878376559327*v(2,0)) + cRHS48*(cRHS226 + 0.49056261216234409*v(0,1) + 0.03522081090086452*v(2,1));
    const double cRHS270 = -cRHS40*(cRHS220 + 0.03522081090086452*v(0,0) + 0.49056261216234409*v(2,0)) + cRHS48*(cRHS147 + cRHS218 + 0.0094373878376559327*v(0,1));
    RHS[0] = -cRHS153*(-cRHS103*cRHS95 + 1.0*cRHS104*cRHS108*cRHS149*rho - cRHS109*cRHS116 + 0.78867513459481287*cRHS119*cRHS123*cRHS151*rho - cRHS124*cRHS128 + 0.21132486540518713*cRHS129*cRHS132*cRHS150*rho - cRHS133*cRHS137 + 1.0*cRHS143*cRHS82*cRHS94*rho - cRHS152 - cRHS27*(cRHS21 + cRHS26) - cRHS36*(cRHS32 + cRHS35) - 0.62200846792814624*cRHS38 - 0.044658198738520456*cRHS39 - cRHS43 + 0.16666666666666669*cRHS54*rho + 0.16666666666666669*cRHS60*rho + 0.62200846792814624*cRHS70*rho + 0.044658198738520456*cRHS76*rho - mu*(-cRHS0*cRHS12 + cRHS14) - mu*(-cRHS0*cRHS5 + cRHS11));
    RHS[1] = -cRHS153*(-cRHS103*cRHS192 + 1.0*cRHS104*cRHS149*cRHS195*rho - cRHS116*cRHS196 + 0.78867513459481287*cRHS119*cRHS151*cRHS199*rho - cRHS128*cRHS200 + 0.21132486540518713*cRHS129*cRHS150*cRHS202*rho - cRHS137*cRHS203 + 1.0*cRHS143*cRHS191*cRHS82*rho - 0.62200846792814624*cRHS177 - 0.044658198738520456*cRHS179 - cRHS181 + 0.16666666666666669*cRHS182*rho + 0.16666666666666669*cRHS183*rho + 0.62200846792814624*cRHS184*rho + 0.044658198738520456*cRHS185*rho - cRHS206 - cRHS27*(cRHS0*cRHS164 + cRHS168) - cRHS36*(cRHS0*cRHS171 + cRHS174) - mu*(cRHS0*cRHS156 - cRHS158*cRHS6) - mu*(cRHS0*cRHS159 - cRHS160*cRHS6));
    RHS[2] = cRHS153*(cRHS100*cRHS127*cRHS237 + 0.044658198738520456*cRHS106 + cRHS114*cRHS136*cRHS232 + 0.16666666666666669*cRHS213 + 0.16666666666666669*cRHS214 + 0.62200846792814624*cRHS216 + 0.044658198738520456*cRHS217 + cRHS221*cRHS222*cRHS224 + cRHS222*cRHS228*cRHS229 + cRHS224*cRHS233 + cRHS229*cRHS234 + cRHS230*cRHS231*cRHS232 + cRHS235*cRHS236*cRHS237 + cRHS238 + cRHS27*(cRHS0*cRHS19 - cRHS212) + cRHS36*(cRHS0*cRHS30 - cRHS215) + cRHS42 + 0.62200846792814624*cRHS86 + mu*(cRHS0*cRHS12 - cRHS211) + mu*(cRHS0*cRHS5 - cRHS209));
    RHS[3] = -cRHS153*(-cRHS101*cRHS127*cRHS192 - cRHS115*cRHS136*cRHS196 - cRHS181 + 0.044658198738520456*cRHS182*rho + 0.62200846792814624*cRHS183*rho + 0.16666666666666669*cRHS184*rho + 0.16666666666666669*cRHS185*rho + cRHS192*cRHS235*cRHS236 + cRHS196*cRHS230*cRHS231 + cRHS200*cRHS221*cRHS222 - cRHS200*cRHS244 + cRHS203*cRHS222*cRHS228 - cRHS203*cRHS245 - 0.62200846792814624*cRHS204 - 0.044658198738520456*cRHS205 - cRHS246 + mu*(cRHS0*cRHS156 + cRHS241*cRHS6) + mu*(cRHS0*cRHS159 + cRHS243*cRHS6) + 0.21132486540518713*mu*(cRHS0*cRHS164 - cRHS174) + 0.78867513459481287*mu*(cRHS0*cRHS171 - cRHS168));
    RHS[4] = -cRHS153*(-cRHS104*cRHS107*cRHS251 + 1.0*cRHS104*cRHS107*cRHS255*rho + 0.21132486540518713*cRHS119*cRHS122*cRHS151*rho - cRHS119*cRHS122*cRHS253 + 0.78867513459481287*cRHS129*cRHS131*cRHS150*rho - cRHS129*cRHS131*cRHS252 - cRHS152 - 0.62200846792814624*cRHS213 - 0.044658198738520456*cRHS214 - 0.16666666666666669*cRHS216 - 0.16666666666666669*cRHS217 - cRHS250*cRHS82*cRHS93 + 1.0*cRHS254*cRHS82*cRHS93*rho - cRHS27*(cRHS0*cRHS30 + cRHS212) - cRHS36*(cRHS0*cRHS19 + cRHS215) - 0.044658198738520456*cRHS38 - 0.62200846792814624*cRHS39 - cRHS42 - mu*(cRHS0*cRHS248 + cRHS209) - mu*(cRHS0*cRHS249 + cRHS211));
    RHS[5] = -cRHS153*(-cRHS102*cRHS260 - cRHS102*cRHS261 + 1.0*cRHS104*cRHS194*cRHS255*rho + 0.21132486540518713*cRHS119*cRHS151*cRHS198*rho - cRHS127*cRHS263 + 0.78867513459481287*cRHS129*cRHS150*cRHS201*rho - cRHS136*cRHS264 - 0.044658198738520456*cRHS177 - 0.62200846792814624*cRHS179 - cRHS180 + 1.0*cRHS190*cRHS254*cRHS82*rho - cRHS206 - cRHS250*cRHS265 - cRHS251*cRHS266 - cRHS252*cRHS267 - cRHS253*cRHS268 - cRHS27*(cRHS0*cRHS170 + cRHS259) - cRHS36*(cRHS0*cRHS163 + cRHS262) - mu*(cRHS0*cRHS257 + cRHS241*cRHS6) - mu*(cRHS0*cRHS258 + cRHS243*cRHS6));
    RHS[6] = -cRHS153*(-cRHS101*cRHS136*cRHS95 - 0.62200846792814624*cRHS106 - cRHS109*cRHS115*cRHS127 - cRHS109*cRHS230*cRHS236 - cRHS124*cRHS222*cRHS269 - cRHS124*cRHS244 - cRHS133*cRHS222*cRHS270 - cRHS133*cRHS245 - cRHS231*cRHS235*cRHS95 - cRHS238 - cRHS43 + 0.62200846792814624*cRHS54*rho + 0.044658198738520456*cRHS60*rho + 0.16666666666666669*cRHS70*rho + 0.16666666666666669*cRHS76*rho - 0.044658198738520456*cRHS86 + 0.78867513459481287*mu*(-cRHS21 + cRHS35) + 0.21132486540518713*mu*(cRHS26 - cRHS32) + mu*(cRHS0*cRHS248 + cRHS11) + mu*(cRHS0*cRHS249 + cRHS14));
    RHS[7] = -cRHS153*(-cRHS100*cRHS136*cRHS265 - cRHS102*cRHS263 - cRHS102*cRHS264 + 0.78867513459481287*cRHS104*cRHS194*cRHS230*rho - cRHS114*cRHS127*cRHS266 + 1.0*cRHS119*cRHS198*cRHS269*rho - cRHS127*cRHS260 + 1.0*cRHS129*cRHS201*cRHS270*rho - cRHS136*cRHS261 - cRHS180 + 0.21132486540518713*cRHS190*cRHS235*cRHS82*rho - 0.044658198738520456*cRHS204 - 0.62200846792814624*cRHS205 - cRHS233*cRHS268 - cRHS234*cRHS267 - cRHS246 + 0.78867513459481287*mu*(cRHS0*cRHS163 - cRHS259) + 0.21132486540518713*mu*(cRHS0*cRHS170 - cRHS262) + mu*(cRHS0*cRHS257 - cRHS158*cRHS6) + mu*(cRHS0*cRHS258 - cRHS160*cRHS6));

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

void IncompressibleNavierStokesQ1P0StructuredElement::GetCellGradientOperator(
    const double a,
    const double b,
    QuadVectorDataView& G)
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
    const HexaVectorDataView& v,
    const double p,
    const HexaVectorDataView& f,
    const HexaVectorDataView& acc,
    std::array<double, 24>& RHS)
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

void IncompressibleNavierStokesQ1P0StructuredElement::GetCellGradientOperator(
    const double a,
    const double b,
    const double c,
    HexaVectorDataView& G)
{

//substitute_G_3d
}
