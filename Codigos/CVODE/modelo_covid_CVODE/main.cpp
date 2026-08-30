#include <stdio.h> 
#include <stdlib.h>
#include <iostream>
#include <cvode/cvode.h>             // Solver das EDOs
#include <nvector/nvector_serial.h>  // nvector permire usar vetores Sundials em modo serial -> sem paralelização
#include <sunmatrix/sunmatrix_dense.h> // Access to dense SUNMatrix
#include <sunlinsol/sunlinsol_dense.h> // Access to dense SUNLinearSolver
#include <sundials/sundials_types.h>  // Definir o tipo do SUNDIALS
#include <sundials/sundials_math.h>   // Funções matemáticas
#include <sundials/sundials_context.h> // SUNContext
#include <vector>
#include <fstream>
#include <sstream>
#include <mpi.h>
#include "de.h"
#include <cmath>
#include <ctime>

#define real double
#define NEQ 16
#define TAM 15
#define epsilon 1e-12

#define RTOL SUN_RCONST(1.0e-4)
#define ATOL SUN_RCONST(1.0e-4)

using realtype = sunrealtype;
using namespace std;

// ==========================================
// Estruturas Globais do SUNDIALS (Reutilizáveis)
// ==========================================
SUNContext sunctx;
N_Vector y_global;
SUNMatrix A_global;
SUNLinearSolver LS_global;
void *cvode_mem_global = NULL;

// ============================
// Condições iniciais
// ============================
real V0 = 61.0, Ap0 = 1.0e6, ApM0 = 0.0, I0 = 0.0, ThN0 = 1.0e6, ThE0 = 0.0, TkN0 = 5.0e5, TkE0 = 0.0, B0 = 2.5e5, Ps0 = 0.0,
    Pl0 = 0.0, Bm0 = 0.0, IgM0 = 0.0, IgG0 = 0.0, C0 = 0.0, NK0 = 1.3e5;
     
// ============================
// Parâmetros
// ============================
real pi_v = 1.47, kv1 = 9.82e-3, kv2 = 6.10e-5, kv3 = 6.45e-2, alpha_ap = 1.0, beta_ap = 1.79e-1, cap1 = 8.0, cap2 = 8.08e6, 
    gamma_apm = 4.0e-2, beta_apm = 1.33e-2, beta_tke= 3.5e-6, alpha_th = 2.17e-4, beta_th = 1.8e-5, pi_th = 1.0e-8, delta_th = 3.0e-1,
    alpha_tk = 1.0, beta_tk = 1.43e-5, pi_tk = 1.0e-8, delta_tk = 3.0e-2, alpha_b = 3.578236584, pi_b1 = 8.98e-5, pi_b2 = 1.27e-8,
    beta_ps = 6.0e-6, beta_pl = 5.0e-6, beta_bm = 1.0e-6, delta_ps = 2.5, delta_pl = 3.5e-1, delta_bm = 9.75e-4, pi_bm1  = 1.0e-5,
    pi_bm2 = 2.5e3, pi_ps = 8.7e-2, pi_pl = 1.0e-3, delta_am = 7.0e-2, delta_ag = 7.0e-2, pi_capm = 3.28e2, pi_ci   = 6.44e-3,
    pi_ctke = 1.78e-2, gamma_c  = 7.04e2, qn = 0.52, dn = 0.07, gamma_ink =  0.000574, gamma_itk = 0.001, pi_cnk = 0.01, Nmax =  3e6;

// ============================
// Vetores Globais
// ============================

// Tempos dos dados experimentais
vector<real> tempo_viremia, tempo_il6, tempo_igg, tempo_igm, tempo_nk, tempo_tcd4, tempo_tcd8, tempo_b;

// Dados Experimentais
vector<real> data_viremia, data_il6, data_igg, data_igm, data_nk, data_tcd4, data_tcd8, data_b;

// Dados em log
vector<real> data_log_viremia, data_log_il6, data_log_igg, data_log_igm, data_log_nk, data_log_tcd4, data_log_tcd8, data_log_b;

// Z-score 
vector<real> z_viremia, z_il6, z_igg, z_igm, z_nk, z_tcd4, z_tcd8, z_b;

// Média e desvio-padrão dos dados
real media_viremia, media_il6, media_igg, media_igm, media_nk, media_tcd4, media_tcd8, media_b;
real dv_viremia, dv_il6, dv_igg, dv_igm, dv_nk, dv_tcd4, dv_tcd8, dv_b;

// Vetores das simulações
vector<real> modelo_tempo;
vector<real> modelo_viremia, modelo_il6, modelo_igg, modelo_igm, modelo_nk, modelo_tcd4, modelo_tcd8, modelo_b;
    
real clip(real valor){
    if(valor < epsilon)    {
        return epsilon;
    }

    return valor;
}

// ============================
// Modelo
// ============================
int f(realtype t, N_Vector y, N_Vector ydot, void *user_data) {

    sunrealtype V = clip(NV_Ith_S(y, 0));
    sunrealtype Ap = clip(NV_Ith_S(y, 1));
    sunrealtype ApM = clip(NV_Ith_S(y, 2));
    sunrealtype I = clip(NV_Ith_S(y, 3));
    sunrealtype ThN = clip(NV_Ith_S(y, 4));
    sunrealtype ThE = clip(NV_Ith_S(y, 5));
    sunrealtype TkN = clip(NV_Ith_S(y, 6));
    sunrealtype TkE = clip(NV_Ith_S(y, 7));
    sunrealtype B = clip(NV_Ith_S(y, 8));
    sunrealtype Ps = clip(NV_Ith_S(y, 9));
    sunrealtype Pl = clip(NV_Ith_S(y, 10));
    sunrealtype Bm = clip(NV_Ith_S(y, 11));
    sunrealtype IgM = clip(NV_Ith_S(y, 12));
    sunrealtype IgG = clip(NV_Ith_S(y, 13));
    sunrealtype C = clip(NV_Ith_S(y, 14));
    sunrealtype Nk = clip(NV_Ith_S(y, 15));

    NV_Ith_S(ydot, 0) = pi_v*V - kv1*V*IgG - kv1*V*IgM - kv2*V*TkE - kv3*V*ApM;
    NV_Ith_S(ydot, 1) = alpha_ap*(C + 1.0)*(Ap0 - Ap) - beta_ap*Ap*((cap1*V)/(cap2 + V)); 
    NV_Ith_S(ydot, 2) = beta_ap*Ap*((cap1*V)/(cap2 + V)) - beta_apm*ApM*V - gamma_apm*ApM;
    NV_Ith_S(ydot, 3) = beta_apm*ApM*V + beta_tke*TkE*V - gamma_apm*I - gamma_ink*Nk*I - gamma_itk*TkE*I;
    NV_Ith_S(ydot, 4) = alpha_th*(ThN0 - ThN) - beta_th*ApM*ThN;
    NV_Ith_S(ydot, 5) = beta_th*ApM*ThN + pi_th*ApM*ThE - delta_th*ThE; 
    NV_Ith_S(ydot, 6) = alpha_tk*(C + 1)*(TkN0 - TkN) - beta_tk*(C + 1)*ApM*TkN; 
    NV_Ith_S(ydot, 7) = beta_tk*(C + 1)*ApM*TkN + pi_tk*ApM*TkE - beta_tke*TkE*V - delta_tk*TkE;
    NV_Ith_S(ydot, 8) = alpha_b*(B0 - B) + pi_b1*V*B + pi_b2*ThE*B - beta_ps*ApM*B - beta_pl*ThE*B - beta_bm*ThE*B; 
    NV_Ith_S(ydot, 9) = beta_ps*ApM*B - delta_ps*Ps; 
    NV_Ith_S(ydot, 10) = beta_pl*ThE*B - delta_pl*Pl + delta_bm*Bm; 
    NV_Ith_S(ydot, 11) = beta_bm*ThE*B + pi_bm1*Bm*(1.0 - (Bm/pi_bm2)) - delta_bm*Bm; 
    NV_Ith_S(ydot, 12) = pi_ps*Ps - delta_am*IgM; 
    NV_Ith_S(ydot, 13) = pi_pl*Pl - delta_ag*IgG; 
    NV_Ith_S(ydot, 14) = pi_capm*ApM + pi_ci*I + pi_ctke*TkE + pi_cnk*Nk - gamma_c*C; 
    NV_Ith_S(ydot, 15) = qn*( Nmax - Nk ) * I - dn * (Nk - NK0); 

    return 0;
}

// ============================
// Funções auxiliares
// ============================


vector<real> transforma_log10(const vector<real>& dados){
    vector<real> dados_log;

    for(int i = 0; i < dados.size(); i++){
        dados_log.push_back(log10(clip(dados[i])));
    }

    return dados_log;
}

real mean(const vector<real>& dados){
    real soma = 0.0;

    for(int i = 0; i < dados.size(); i++){
        soma += dados[i];
    }

    return soma / dados.size();
}

real stddev(const vector<real>& dados){
    real media = mean(dados);
    real soma = 0.0;

    for(int i = 0; i < dados.size(); i++){
        soma += pow(dados[i] - media, 2);
    }

    return sqrt(soma / dados.size());
}

vector<real> calcula_zscore(const vector<real>& dados, real media, real dv){
    vector<real> z;

    for(int i = 0; i < dados.size(); i++){
        z.push_back((dados[i] - media) / dv);
    }

    return z;
}

real mse(const vector<real>& modelo_z, const vector<real>& dados_z){
    if(modelo_z.size() != dados_z.size()){
        return 1e18;
    }

    real soma = 0.0;
    vector<real> res;
    for(int i = 0; i < dados_z.size(); i++){
        res.push_back(pow((modelo_z[i] - dados_z[i]), 2));
    }

    return mean(res);
}

// ============================
// Interpolação
// ============================
static real lagrange(real x, vector<real>& x_v, vector<real>& y_v)
{
    if (x <= x_v[0]) return y_v[0];
    if (x >= x_v[x_v.size() - 1]) return y_v[y_v.size() - 1];

    real x0, x1, y0, y1, L0, L1;

    for(int i = 1; i < x_v.size(); i++)
    {
        if(x <= x_v[i])
        {
            x0 = x_v[i - 1];
            x1 = x_v[i];
            y0 = y_v[i - 1];
            y1 = y_v[i];
            break;
        }
    }

    L0 = (x - x1) / (x0 - x1);
    L1 = (x - x0) / (x1 - x0);

    return L0 * y0 + L1 * y1;
}

// ============================
// Lendo Dados Experimentais
// ============================
void ler_csv(const string& nome_arquivo, vector<real>& tempo, vector<real>& valor, real fator){
    
    ifstream arquivo(nome_arquivo);

    if (!arquivo.is_open()){
        cout << "Erro ao abrir " << nome_arquivo << endl;
        exit(EXIT_FAILURE);
    }

    string linha;
    getline(arquivo, linha); // Ignora o cabeçalho

    while (getline(arquivo, linha)){
        if (linha.empty())
            continue;

        stringstream ss(linha);
        vector<string> campos;
        string campo;

        while (getline(ss, campo, ',')){
            // Verificando se tem aquele r no final da linha por causa do windows
            if (!campo.empty() && campo.back() == '\r')
                campo.pop_back();

            campos.push_back(campo);
        }

        // Pega a ultima coluna, type
        string type = campos[campos.size() - 1];

        if (type == "mean"){
            real x = stod(campos[0]);
            real y = stod(campos[1]);

            tempo.push_back(x + 4.15);
            valor.push_back(y * fator);

        }
    }

    arquivo.close();
}

real calcula_erro_populacao(const vector<real>& tempo_dados, const vector<real>& dados_z, const vector<real>& tempo_modelo,
    const vector<real>& modelo, real media_dados, real dv_dados){

    real soma_mse = 0.0;

    for(int i = 0; i < tempo_dados.size(); i++){

        real valor = lagrange(tempo_dados[i],
                              const_cast<vector<real>&>(tempo_modelo),
                              const_cast<vector<real>&>(modelo));

        real valor_log = log10(clip(valor));

        real valor_log_z = (valor_log - media_dados) / dv_dados;

        real diferenca = valor_log_z - dados_z[i];
        soma_mse += diferenca*diferenca;
    }
    return soma_mse / tempo_dados.size();
}


// VERIFICANDO ERROS DO SUNDIALS
static int check_retval(void* returnvalue, const char* funcname, int opt){
  int* retval;

  // Caso para quando a função deveria retornar um ponteiro
  if (opt == 0 && returnvalue == NULL){
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return (1);
	// stderr = saída padrão de erro
  }

  // caso para quando a função SUNDIAL retorn um inteiro de status
  /* 
  normalmente: 
  	retval >= 0  → sucesso ou aviso
	retval < 0   → erro
  */
  else if (opt == 1){
    retval = (int*)returnvalue;
    if (*retval < 0){
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with retval = %d\n\n",
              funcname, *retval);
      return (1);
    }
  }

  // caso para verificar se o ponteiro também é null, normalmente para funções qque alocam memória fora do SUNDIALS
  else if (opt == 2 && returnvalue == NULL){
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return (1);
  }

  return (0);
}


// ============================
// Função objetivo
// ============================
static real model(real *x){

    modelo_tempo.clear();
    modelo_tcd4.clear();
    modelo_tcd8.clear();
    modelo_viremia.clear();
    modelo_il6.clear();
    modelo_igg.clear();
    modelo_igm.clear();
    modelo_b.clear();
    modelo_nk.clear();

    // parâmetros ajustados
    pi_v     = x[0];
    kv3      = x[1];
    beta_ap  = x[2];
    cap1     = x[3];
    cap2     = x[4];
    beta_apm = x[5];
    beta_th  = x[6];
    delta_th = x[7];
    beta_tk  = x[8];
    gamma_c  = x[9];
    dn       = x[10];
    pi_cnk   = x[11];
    Ap0      = x[12];
    TkN0     = x[13];
    B0       = x[14];

    real Ap0_y = 1.0e6, TkN0_y = 5.0e5, B0_y = 2.5e5;

    NV_Ith_S(y_global, 0)  = V0;
    NV_Ith_S(y_global, 1)  = Ap0_y;
    NV_Ith_S(y_global, 2)  = ApM0;
    NV_Ith_S(y_global, 3)  = I0;
    NV_Ith_S(y_global, 4)  = ThN0;
    NV_Ith_S(y_global, 5)  = ThE0;
    NV_Ith_S(y_global, 6)  = TkN0_y;
    NV_Ith_S(y_global, 7)  = TkE0;
    NV_Ith_S(y_global, 8)  = B0_y;
    NV_Ith_S(y_global, 9)  = Ps0;
    NV_Ith_S(y_global, 10) = Pl0;
    NV_Ith_S(y_global, 11) = Bm0;
    NV_Ith_S(y_global, 12) = IgM0;
    NV_Ith_S(y_global, 13) = IgG0;
    NV_Ith_S(y_global, 14) = C0;
    NV_Ith_S(y_global, 15) = NK0;   


    // Se o CVODE não for criado corretamente
    if (cvode_mem_global == NULL) {
        printf("Error: CVodeCreate failed\n");
        return 1e18;
    }
    
    realtype t0 = 0.0;
    realtype t = t0;     // Tempo inicial
    realtype t1 = 72.15; // Tempo final
    realtype dt = 0.1;   // Passo de tempo

    int status = CVodeReInit(cvode_mem_global, t0, y_global);
    if (status != CV_SUCCESS) {
        // Se falhar na reinicialização, retorna o erro alto para penalizar o indivíduo
        return 1e18; 
    }

    while (t < t1) {
        realtype tret;

        int retval = CVode(cvode_mem_global, t + dt, y_global, &tret, CV_NORMAL);

        if(retval < 0){
            return 1e18;
        }

        t = tret; 

        modelo_tempo.push_back(t);
        modelo_viremia.push_back(NV_Ith_S(y_global, 0));
        modelo_igm.push_back(NV_Ith_S(y_global, 12));
        modelo_igg.push_back(NV_Ith_S(y_global, 13));
        modelo_il6.push_back(NV_Ith_S(y_global, 14));
        modelo_nk.push_back(NV_Ith_S(y_global, 15));
        modelo_tcd4.push_back(NV_Ith_S(y_global, 5));
        modelo_tcd8.push_back(NV_Ith_S(y_global, 7));
        modelo_b.push_back(NV_Ith_S(y_global, 8) + NV_Ith_S(y_global, 11));
        
    }

    real erro_viremia = calcula_erro_populacao(
        tempo_viremia,
        z_viremia,
        modelo_tempo,
        modelo_viremia,
        media_viremia,
        dv_viremia
    );
    real erro_il6 = calcula_erro_populacao(
        tempo_il6,
        z_il6,
        modelo_tempo,
        modelo_il6,
        media_il6,
        dv_il6
    );
    real erro_igm = calcula_erro_populacao(
        tempo_igm,
        z_igm,
        modelo_tempo,
        modelo_igm,
        media_igm,
        dv_igm
    );
    real erro_igg = calcula_erro_populacao(
        tempo_igg,
        z_igg,
        modelo_tempo,
        modelo_igg,
        media_igg,
        dv_igg
    );
    real erro_tcd4 = calcula_erro_populacao(
        tempo_tcd4,
        z_tcd4,
        modelo_tempo,
        modelo_tcd4,
        media_tcd4,
        dv_tcd4
    );
    real erro_tcd8 = calcula_erro_populacao(
        tempo_tcd8,
        z_tcd8,
        modelo_tempo,
        modelo_tcd8,
        media_tcd8,
        dv_tcd8
    );
    real erro_nk = calcula_erro_populacao(
        tempo_nk,
        z_nk,
        modelo_tempo,
        modelo_nk,
        media_nk,
        dv_nk
    );
    real erro_b = calcula_erro_populacao(
        tempo_b,
        z_b,
        modelo_tempo,
        modelo_b,
        media_b,
        dv_b
    );

    return erro_il6 +  erro_viremia + erro_igg + erro_igm;// + erro_nk + erro_tcd4 + erro_tcd8 + erro_b;
}


// ===============================
//  Salvar simulação com melhor x 
// ===============================
static void salvar_resultados_csv(real *x, const string& nome_arquivo){
    
    // Usa o melhor vetor encontrado pelo DE
    pi_v     = x[0];
    kv3      = x[1];
    beta_ap  = x[2];
    cap1     = x[3];
    cap2     = x[4];
    beta_apm = x[5];
    beta_th  = x[6];
    delta_th = x[7];
    beta_tk  = x[8];
    gamma_c  = x[9];
    dn       = x[10];
    pi_cnk   = x[11];
    Ap0      = x[12];
    TkN0     = x[13];
    B0       = x[14];

    SUNContext sunctx;
    SUNContext_Create(MPI_COMM_WORLD, &sunctx);

    N_Vector y = N_VNew_Serial(NEQ, sunctx);

    real Ap0_y = 1.0e6, TkN0_y = 5.0e5, B0_y = 2.5e5;
    
    NV_Ith_S(y, 0)  = V0;
    NV_Ith_S(y, 1)  = Ap0_y;
    NV_Ith_S(y, 2)  = ApM0;
    NV_Ith_S(y, 3)  = I0;
    NV_Ith_S(y, 4)  = ThN0;
    NV_Ith_S(y, 5)  = ThE0;
    NV_Ith_S(y, 6)  = TkN0_y;
    NV_Ith_S(y, 7)  = TkE0;
    NV_Ith_S(y, 8)  = B0_y;
    NV_Ith_S(y, 9)  = Ps0;
    NV_Ith_S(y, 10) = Pl0;
    NV_Ith_S(y, 11) = Bm0;
    NV_Ith_S(y, 12) = IgM0;
    NV_Ith_S(y, 13) = IgG0;
    NV_Ith_S(y, 14) = C0;
    NV_Ith_S(y, 15) = NK0;

    void *cvode_mem = CVodeCreate(CV_BDF, sunctx);
    if (cvode_mem == NULL) {
        cout << "Erro: CVodeCreate falhou ao salvar CSV." << endl;
        N_VDestroy(y);
        SUNContext_Free(&sunctx);
        return;
    }

    realtype t0 = 0.0;
    int retval = CVodeInit(cvode_mem, f, t0, y);
    if (retval != CV_SUCCESS) {
        cout << "Erro: CVodeInit falhou ao salvar CSV." << endl;
        N_VDestroy(y);
        CVodeFree(&cvode_mem);
        SUNContext_Free(&sunctx);
        return;
    }

    retval = CVodeSStolerances(cvode_mem, RTOL, ATOL);
    if (retval != CV_SUCCESS) {
        cout << "Erro: CVodeSStolerances falhou ao salvar CSV." << endl;
        N_VDestroy(y);
        CVodeFree(&cvode_mem);
        SUNContext_Free(&sunctx);
        return;
    }

    SUNMatrix A = SUNDenseMatrix(NEQ, NEQ, sunctx);
    SUNLinearSolver LS = SUNLinSol_Dense(y, A, sunctx);

    retval = CVodeSetLinearSolver(cvode_mem, LS, A);
    if (retval != CV_SUCCESS) {
        cout << "Erro: CVodeSetLinearSolver falhou ao salvar CSV." << endl;
        N_VDestroy(y);
        SUNLinSolFree(LS);
        SUNMatDestroy(A);
        CVodeFree(&cvode_mem);
        SUNContext_Free(&sunctx);
        return;
    }

    ofstream csv(nome_arquivo);

    if (!csv.is_open()) {
        cout << "Erro ao criar arquivo " << nome_arquivo << endl;
        N_VDestroy(y);
        SUNLinSolFree(LS);
        SUNMatDestroy(A);
        CVodeFree(&cvode_mem);
        SUNContext_Free(&sunctx);
        return;
    }

    csv << "Time,V,Ap,ApM,I,ThN,ThE,TkN,TkE,B,Ps,Pl,Bm,IgM,IgG,C,NK\n";

    realtype t = t0;
    realtype t1 = 72.15;
    realtype dt = 0.1;

    // Salva também a condição inicial
    csv << t << ","
        << NV_Ith_S(y, 0)  << "," << NV_Ith_S(y, 1)  << "," << NV_Ith_S(y, 2)  << ","
        << NV_Ith_S(y, 3)  << "," << NV_Ith_S(y, 4)  << "," << NV_Ith_S(y, 5)  << ","
        << NV_Ith_S(y, 6)  << "," << NV_Ith_S(y, 7)  << "," << NV_Ith_S(y, 8)  << ","
        << NV_Ith_S(y, 9)  << "," << NV_Ith_S(y, 10) << "," << NV_Ith_S(y, 11) << ","
        << NV_Ith_S(y, 12) << "," << NV_Ith_S(y, 13) << "," << NV_Ith_S(y, 14) << ","
        << NV_Ith_S(y, 15) << "\n";

    while (t < t1) {
        realtype tret;
        retval = CVode(cvode_mem, t + dt, y, &tret, CV_NORMAL);

        if (retval < 0) {
            cout << "Erro: CVode falhou ao salvar CSV em t = " << t << endl;
            break;
        }

        t = tret;

        csv << t << ","
            << NV_Ith_S(y, 0)  << "," << NV_Ith_S(y, 1)  << "," << NV_Ith_S(y, 2)  << ","
            << NV_Ith_S(y, 3)  << "," << NV_Ith_S(y, 4)  << "," << NV_Ith_S(y, 5)  << ","
            << NV_Ith_S(y, 6)  << "," << NV_Ith_S(y, 7)  << "," << NV_Ith_S(y, 8)  << ","
            << NV_Ith_S(y, 9)  << "," << NV_Ith_S(y, 10) << "," << NV_Ith_S(y, 11) << ","
            << NV_Ith_S(y, 12) << "," << NV_Ith_S(y, 13) << "," << NV_Ith_S(y, 14) << ","
            << NV_Ith_S(y, 15) << "\n";
    }

    csv.close();

    N_VDestroy(y);
    SUNLinSolFree(LS);
    SUNMatDestroy(A);
    CVodeFree(&cvode_mem);
    SUNContext_Free(&sunctx);
}


int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    modelo_tempo.reserve(750);
    modelo_tcd4.reserve(750);
    modelo_tcd8.reserve(750);
    modelo_viremia.reserve(750);
    modelo_il6.reserve(750);
    modelo_igg.reserve(750);
    modelo_igm.reserve(750);
    modelo_b.reserve(750);
    modelo_nk.reserve(750);

    ler_csv("/mnt/c/Users/mique/OneDrive/Documentos/UFJF/Modelagem Fisiologica/Data/covid_moderado/TCD4_covid_moderado.csv", tempo_tcd4, data_tcd4, 1.0/1000.0);
    data_log_tcd4 = transforma_log10(data_tcd4);
    media_tcd4 = mean(data_log_tcd4);
    dv_tcd4 = stddev(data_log_tcd4);
    z_tcd4 = calcula_zscore(data_log_tcd4, media_tcd4, dv_tcd4);

    ler_csv("/mnt/c/Users/mique/OneDrive/Documentos/UFJF/Modelagem Fisiologica/Data/covid_moderado/TCD8_covid_moderado.csv", tempo_tcd8, data_tcd8, 1.0/1000.0);
    data_log_tcd8 = transforma_log10(data_tcd8);
    media_tcd8 = mean(data_log_tcd8);
    dv_tcd8 = stddev(data_log_tcd8);
    z_tcd8 = calcula_zscore(data_log_tcd8, media_tcd8, dv_tcd8);

    ler_csv("/mnt/c/Users/mique/OneDrive/Documentos/UFJF/Modelagem Fisiologica/Data/viremia_covid.csv", tempo_viremia, data_viremia, 1);
    data_log_viremia = transforma_log10(data_viremia);
    media_viremia = mean(data_log_viremia);
    dv_viremia = stddev(data_log_viremia);
    z_viremia = calcula_zscore(data_log_viremia, media_viremia, dv_viremia);
    
    ler_csv("/mnt/c/Users/mique/OneDrive/Documentos/UFJF/Modelagem Fisiologica/Data/citocinas_covid.csv", tempo_il6, data_il6, 1);
    data_log_il6 = transforma_log10(data_il6);
    media_il6 = mean(data_log_il6);
    dv_il6 = stddev(data_log_il6);
    z_il6 = calcula_zscore(data_log_il6, media_il6, dv_il6);
    
    ler_csv("/mnt/c/Users/mique/OneDrive/Documentos/UFJF/Modelagem Fisiologica/Data/covid_moderado/NK_covid_moderado.csv", tempo_nk, data_nk, 1.0/1000.0);
    data_log_nk = transforma_log10(data_nk);
    media_nk = mean(data_log_nk);
    dv_nk = stddev(data_log_nk);
    z_nk = calcula_zscore(data_log_nk, media_nk, dv_nk);
    
    ler_csv("/mnt/c/Users/mique/OneDrive/Documentos/UFJF/Modelagem Fisiologica/Data/covid_moderado/Bcell_covid_moderado.csv", tempo_b, data_b, 1.0/1000.0);
    data_log_b = transforma_log10(data_b);
    media_b = mean(data_log_b);
    dv_b = stddev(data_log_b);
    z_b = calcula_zscore(data_log_b, media_b, dv_b);
    
    ler_csv("/mnt/c/Users/mique/OneDrive/Documentos/UFJF/Modelagem Fisiologica/Data/covid_moderado/IgM_covid_moderado.csv", tempo_igm, data_igm, 1.0);
    data_log_igm = transforma_log10(data_igm);
    media_igm = mean(data_log_igm);
    dv_igm = stddev(data_log_igm);
    z_igm = calcula_zscore(data_log_igm, media_igm, dv_igm);
    
    ler_csv("/mnt/c/Users/mique/OneDrive/Documentos/UFJF/Modelagem Fisiologica/Data/covid_moderado/IgG_covid_moderado.csv", tempo_igg, data_igg, 1.0);
    data_log_igg = transforma_log10(data_igg);
    media_igg = mean(data_log_igg);
    dv_igg = stddev(data_log_igg);
    z_igg = calcula_zscore(data_log_igg, media_igg, dv_igg);
    
    /*real parametros_teste[15] = {
        0.640143,
        0.0125757,
        0.194366,
        4.26304,
        6.11478e+06,
        0.0152946,
        8.24241e-05,
        0.260704,
        3.30689e-06,
        916.519,
        9.35827e-05,
        0.00132746,
        974389,
        115332,
        341491
    };*/

    /*real parametros_teste[15] = {
        0.9,
        0.0248544,
        0.424335,
        10.7653,
        5.37473e+06,
        0.0395011,
        2.86424e-05,
        0.243894,
        1.18669e-05,
        1004.49,
        0.0540172,
        0.00101818,
        499268,
        202227,
        510536
    };

    salvar_resultados_csv(parametros_teste, "/mnt/c/Users/mique/OneDrive/Documentos/UFJF/Modelagem Fisiologica/Codigos/CVODE/modelo_covid_CVODE/Resultados_simulacoes/teste.csv");
*/

    // 2. Inicialização das Memórias SUNDIALS
    SUNContext_Create(MPI_COMM_WORLD, &sunctx);
    y_global = N_VNew_Serial(NEQ, sunctx);
    cvode_mem_global = CVodeCreate(CV_BDF, sunctx);
    
    // Inicialização do CVODE
    realtype t0_init = 0.0;
    NV_Ith_S(y_global, 0) = V0;
    CVodeInit(cvode_mem_global, f, t0_init, y_global);
    CVodeSStolerances(cvode_mem_global, RTOL, ATOL);
    
    A_global = SUNDenseMatrix(NEQ, NEQ, sunctx);
    LS_global = SUNLinSol_Dense(y_global, A_global, sunctx);
    CVodeSetLinearSolver(cvode_mem_global, LS_global, A_global);
    
    CVodeSetMaxNumSteps(cvode_mem_global, 2000);
    
    /*vector<real> l_bounds, h_bounds;
    
    real low = 0.45,  high = 1.2;
    
    l_bounds.push_back(pi_v * low);
    l_bounds.push_back(kv3 * low);
    l_bounds.push_back(beta_ap * low);
    l_bounds.push_back(cap1 * low);
    l_bounds.push_back(cap2 * low);
    l_bounds.push_back(beta_apm * low);
    l_bounds.push_back(beta_th * low);
    l_bounds.push_back(delta_th * low);
    l_bounds.push_back(beta_tk * low);
    l_bounds.push_back(gamma_c * low);
    l_bounds.push_back(dn * low);
    l_bounds.push_back(pi_cnk * low);
    l_bounds.push_back(Ap0 * low);
    l_bounds.push_back(TkN0 * low);
    l_bounds.push_back(B0 * low);
    
    h_bounds.push_back(pi_v * high);
    h_bounds.push_back(kv3 * high);
    h_bounds.push_back(beta_ap * high);
    h_bounds.push_back(cap1 * high);
    h_bounds.push_back(cap2 * high);
    h_bounds.push_back(beta_apm * high);
    h_bounds.push_back(beta_th * high);
    h_bounds.push_back(delta_th * high);
    h_bounds.push_back(beta_tk * high);
    h_bounds.push_back(gamma_c * high);
    h_bounds.push_back(dn * high);
    h_bounds.push_back(pi_cnk * high);
    h_bounds.push_back(Ap0 * high);
    h_bounds.push_back(TkN0 * high);
    h_bounds.push_back(B0 * high);*/

    vector<real> l_bounds = {
        0.7,
        0.009,
        0.1,
        5,
        5e6,
        0.01,
        7e-5,
        0.1,
        2e-6,
        1000,
        0.0001,
        0.001,
        900000,
        100000,
        400000

    };
    
    vector<real> h_bounds = {
        1.0,
        0.02,
        0.4,
        8,
        9e6,
        0.04,
        7e-4,
        0.5,
        5e-6,
        1300,
        0.001,
        0.003,
        1e6,
        200000,
        600000
    };

    /*vector<real> l_bounds, h_bounds;

    for(int i = 0; i < TAM; i++){
        l_bounds.push_back(parametros_teste[i] * 0.9);
        h_bounds.push_back(parametros_teste[i] * 1.5);
    }*/

    ofstream lower_bounds("/mnt/c/Users/mique/OneDrive/Documentos/UFJF/Modelagem Fisiologica/Codigos/CVODE/Lower_bounds/teste.csv");
    ofstream higher_bounds("/mnt/c/Users/mique/OneDrive/Documentos/UFJF/Modelagem Fisiologica/Codigos/CVODE/Higher_bounds/teste.csv");
    
    for(int i = 0; i < 15; i++) {
        lower_bounds << l_bounds[i] << endl;
    }

    for(int i = 0; i < 15; i++) {
        higher_bounds << h_bounds[i] << endl;
    }
        
    int n_gens = 1000;
    int pop_size = 150;
    int num_par = TAM;
    
    real *x = differential_evolution(
        false,
        model,
        &l_bounds[0],
        &h_bounds[0],
        n_gens,
        0.7,
        0.5,
        1.0, 
        pop_size,
        num_par,
        BEST1BIN,
        time(NULL),
        0,
        0.01
    );

    ofstream params_otimos("/mnt/c/Users/mique/OneDrive/Documentos/UFJF/Modelagem Fisiologica/Codigos/params_otimos_cvode/teste.csv");
    
    cout << "Parametros otimos:" << endl;

    for(int i = 0; i < num_par; i++)
    {
        cout << i << ": " << x[i] << endl;
        params_otimos << x[i] << endl;
    }

    cout << "Erro final: " << model(x) << endl;

    salvar_resultados_csv(x, "/mnt/c/Users/mique/OneDrive/Documentos/UFJF/Modelagem Fisiologica/Codigos/CVODE/modelo_covid_CVODE/Resultados_simulacoes/teste.csv");
    
    N_VDestroy(y_global);
    SUNLinSolFree(LS_global);
    SUNMatDestroy(A_global);
    CVodeFree(&cvode_mem_global);
    SUNContext_Free(&sunctx);
    MPI_Finalize();
    free(x);

    return 0;
}