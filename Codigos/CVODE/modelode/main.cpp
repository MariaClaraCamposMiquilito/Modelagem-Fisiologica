#include "de.h" // Implementação do DE
#include <math.h>
#include <cstdio>
#include <ctime>
#include <cvode/cvode.h>            
#include <nvector/nvector_serial.h>             
#include <stdio.h>
#include <stdlib.h>
#include <sunlinsol/sunlinsol_dense.h> 
#include <sunmatrix/sunmatrix_dense.h>             
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>  
#include <cmath>
#include <iostream>

#ifndef SUN_COMM_NULL
#define SUN_COMM_NULL NULL
#endif

#define TAM 12 // Número de parâmetros

#define NEQ   12  // Número de EDOs

// Tolerâncias
#define RTOL  SUN_RCONST(1.0e-4)           
#define ATOL  SUN_RCONST(1.0e-4)

#define ZERO SUN_RCONST(0.0)

using namespace std;


// condições iniciais globais
real V0 = 2.38303916e+05, Ap0 = 1e6, Apm0 = 0.0, Thn0 = 1e6,
	 The0 = 0.0, Tkn0 = 5.0e5, Tke0 = 0.0, B0 = 2.5e5,
	 Ps0 = 0.0, Pl0 = 0.0, Bm0 = 0.0, A0 = 0.0;

// parâmetros
real pi_v = 1.10896847e-2, c_v1 = 1.12583762e-9, c_v2 = 6.0e-1, k_v1 = 2.07531796e-5,
	 k_v2 = 9.86261312e-4, alfa_ap = 5.13781171e-1, beta_ap = 4.34898450e5, c_ap1 = 7.28737138e0,
	 c_ap2 = 9.20346004e6, delta_apm = 8.34917439e4, alfa_th = 2.17e-4, beta_th = 9.64771407e-4,
	 pi_th = 3.50989237e-3, delta_th = 4.35459566e-2, alfa_tk = 2.41095439e0, beta_tk = 5.76578454e-6,
	 pi_tk = 4.43826666e-9, delta_tk = 1.29520278e-5, alfa_b = 3.58e2, pi_b1 = 8.98e-5,
	 pi_b2 = 1.27e-8, beta_ps = 6.0e-6, beta_pl = 5.0e-6, beta_bm = 1.0e-6,
	 delta_ps = 2.5e0, delta_pl = 3.5e-1, gamma_bm = 9.75e-4, pi_bm1 = 8.11167858e0,
	 pi_bm2 = 3.79652355e3, pi_ps = 4.00409160e4, pi_pl = 1.41972397e4, delta_a = 3.53244981e5;

// Vetores globais -> guardam os dados lidos do csv
vector<real> cd4_x, cd4_y, cd8_x, cd8_y, virus_x, virus_y, ant_x, ant_y;
vector<real> cd4_y_v, cd8_y_v, virus_y_v, ant_y_v;

// Guardam os índices inicial e final de cada conjunto de dados dentro do vetor de tempo do modelo.
int cd4_i, cd4_f, cd8_i, cd8_f, virus_i, virus_f, ant_i, ant_f;

static int f(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data)
{
	/*
    // condições iniciais
    sunrealtype Ap0 = 1e6, Thn0 = 1e6, Tkn0 = 5.0e5, B0 = 2.5e5;

    // parâmetros
    sunrealtype pi_v = 1.10896847e-2, c_v1 = 1.12583762e-9, c_v2 = 6.0e-1, k_v1 = 2.07531796e-5,
                k_v2 = 9.86261312e-4, alfa_ap = 5.13781171e-1, beta_ap = 4.34898450e5, c_ap1 = 7.28737138e0,
                c_ap2 = 9.20346004e6, delta_apm = 8.34917439e4, alfa_th = 2.17e-4, beta_th = 9.64771407e-4,
                pi_th = 3.50989237e-3, delta_th = 4.35459566e-2, alfa_tk = 2.41095439e0, beta_tk = 5.76578454e-6,
                pi_tk = 4.43826666e-9, delta_tk = 1.29520278e-5, alfa_b = 3.58e2, pi_b1 = 8.98e-5,
                pi_b2 = 1.27e-8, beta_ps = 6.0e-6, beta_pl = 5.0e-6, beta_bm = 1.0e-6,
                delta_ps = 2.5e0, delta_pl = 3.5e-1, gamma_bm = 9.75e-4, pi_bm1 = 8.11167858e0,
                pi_bm2 = 3.79652355e3, pi_ps = 4.00409160e4, pi_pl = 1.41972397e4, delta_a = 3.53244981e5;
	*/
    sunrealtype V = NV_Ith_S(y, 0);
    sunrealtype Ap = NV_Ith_S(y, 1);
    sunrealtype Apm = NV_Ith_S(y, 2);
    sunrealtype Thn = NV_Ith_S(y, 3);
    sunrealtype The = NV_Ith_S(y, 4);
    sunrealtype Tkn = NV_Ith_S(y, 5);
    sunrealtype Tke = NV_Ith_S(y, 6);
    sunrealtype B = NV_Ith_S(y, 7);
    sunrealtype Ps = NV_Ith_S(y, 8);
    sunrealtype Pl = NV_Ith_S(y, 9);
    sunrealtype Bm = NV_Ith_S(y, 10);
    sunrealtype A = NV_Ith_S(y, 11);

    /*dVdt =*/ NV_Ith_S(ydot, 0) = pi_v * V - ((c_v1 * V) / (c_v2 + V)) - k_v1 * V * A - k_v2 * V * Tke; // vírus
    
    /*dApdt =*/ NV_Ith_S(ydot, 1) = alfa_ap * (Ap0 - Ap) - beta_ap * Ap * ((c_ap1 * V) / (c_ap2 + V));  // APCs imaturas
    
    /*dApmdt =*/ NV_Ith_S(ydot, 2) = beta_ap * Ap * ((c_ap1 * V) / (c_ap2 + V)) - delta_apm * Apm;  // APCs maduras

    /*dThndt =*/ NV_Ith_S(ydot, 3) = alfa_th * (Thn0 - Thn) - beta_th * Apm * Thn;  // células T CD4+ imaturas

    /*dThedt =*/ NV_Ith_S(ydot, 4) = beta_th * Apm * Thn + pi_th * Apm * The - delta_th * The; // células T CD4+ maduras

    /*dTkndt =*/ NV_Ith_S(ydot, 5) = alfa_tk * (Tkn0 - Tkn) - beta_tk * Apm * Tkn; // células T CD8+ imaturas

    /*dTkedt =*/ NV_Ith_S(ydot, 6) = beta_tk * Apm * Tkn + pi_tk * Apm * Tke - delta_tk * Tke; // células T CD8+ maduras

    /*dBdt =*/ NV_Ith_S(ydot, 7) = alfa_b * (B0 - B) + pi_b1 * V * B + pi_b2 * The * B - beta_ps * Apm * B - beta_pl * The * B - beta_bm * The * B; // células B

    /*dPsdt =*/ NV_Ith_S(ydot, 8) = beta_ps * Apm * B - delta_ps * Ps; // plasmócitos de vida curta

    /*dPldt =*/ NV_Ith_S(ydot, 9) = beta_pl * The * B - delta_pl * Pl + gamma_bm * Bm; // plasmócitos de vida longa

    /*dBmdt =*/ NV_Ith_S(ydot, 10) = beta_bm * The * B + pi_bm1 * Bm * (1 - (Bm / pi_bm2)) - gamma_bm * Bm; // células B de memória

    /*dAdt =*/ NV_Ith_S(ydot, 11) = pi_ps * Ps + pi_pl * Pl - delta_a * A; // anticorpos

    return 0;
}

// Essa função verifica se alguma chamada do SUNDIALS deu erro.
// recebe o retorno da função que você quer verificar, o nome da função, para imprimir e identifcar qual falhou
// e recebe opt, o tipo de verificação que será feita -> 
static int check_retval(void* returnvalue, const char* funcname, int opt)
{
  int* retval;

  // Caso para quando a função deveria retornar um ponteiro
  if (opt == 0 && returnvalue == NULL)
  {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return (1);
	// stderr = saída padrão de erro
  }

  // casp para quando a função SUNDIAL retorn um inteiro de status
  /* 
  normalmente: 
  	retval >= 0  → sucesso ou aviso
	retval < 0   → erro
  */
  else if (opt == 1)
  {
    retval = (int*)returnvalue;
    if (*retval < 0)
    {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with retval = %d\n\n",
              funcname, *retval);
      return (1);
    }
  }

  // caso para verificar se o ponteiro também é null, normalmente para funções qque alocam memória fora do SUNDIALS
  else if (opt == 2 && returnvalue == NULL)
  {
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return (1);
  }

  return (0);
}

// x => vetor de parâmetros
// Serve para salvar a simulação final -> recebe o melhor vetor x, roda o CVODE e guarda no csv
static real modelcsv(real *x)
{
	SUNContext sunctx;
	sunrealtype t0, tf;
	N_Vector y;
	N_Vector abstol;
	SUNMatrix A;
	SUNLinearSolver LS;
	void* cvode_mem;
	int retval;

	y         = NULL;
	abstol    = NULL;
	A         = NULL;
	LS        = NULL;
	cvode_mem = NULL;

	t0 = 0.0;
	tf = 60.0;

	retval = SUNContext_Create(SUN_COMM_NULL, &sunctx);
	if (check_retval(&retval, "SUNContext_Create", 1)) { return (1); }

	y = N_VNew_Serial(NEQ, sunctx);
	if (check_retval((void*)y, "N_VNew_Serial", 0)) { return (1); }

	/*
    Params description:

    x[0]=> V0
    x[1]=> pi_v
    x[2]=> beta_tk
    x[3]=> k_v2
    x[4]=> beta_ap
    x[5]=> delta_apm
    x[6]=> pi_tk
    x[7]=> delta_tk
    x[8]=> delta_a
    x[9]=> alfa_tk
    x[10]=> pi_th
    x[11]=> delta_th
    */
	
	//NV_Ith_S(y, 0) = 2.38303916e+05;
	NV_Ith_S(y, 0) = x[0];
	NV_Ith_S(y, 1) = 1e6;
	NV_Ith_S(y, 2) = 0.0;
	NV_Ith_S(y, 3) = 1e6;
	NV_Ith_S(y, 4) = 0.0;
	NV_Ith_S(y, 5) = 5.0e5;
	NV_Ith_S(y, 6) = 0.0;
	NV_Ith_S(y, 7) = 2.5e5;
	NV_Ith_S(y, 8) = 0.0;
	NV_Ith_S(y, 9) = 0.0;
	NV_Ith_S(y, 10) = 0.0;
	NV_Ith_S(y, 11) = 0.0;

	pi_v = x[1];
    beta_tk = x[2];
    k_v2 = x[3];
    beta_ap = x[4];
    delta_apm = x[5];
    pi_tk = x[6];
    delta_tk = x[7];
    delta_a = x[8];
    alfa_tk = x[9];
    pi_th = x[10];
    delta_th = x[11];

	abstol = N_VNew_Serial(NEQ, sunctx);
	if (check_retval((void*)abstol, "N_VNew_Serial", 0)) { return (1); }

	cvode_mem = CVodeCreate(CV_BDF, sunctx);
  	if (check_retval((void*)cvode_mem, "CVodeCreate", 0)) { return (1); }

	retval = CVodeInit(cvode_mem, f, t0, y);
  	if (check_retval(&retval, "CVodeInit", 1)) { return (1); }

	retval = CVodeSStolerances(cvode_mem, RTOL, ATOL);
    if (check_retval(&retval, "CVodeSStolerances", 1)) { return (1); }

	A = SUNDenseMatrix(NEQ, NEQ, sunctx);
	if (check_retval((void*)A, "SUNDenseMatrix", 0)) { return (1); }

	LS = SUNLinSol_Dense(y, A, sunctx);
	if (check_retval((void*)LS, "SUNLinSol_Dense", 0)) { return (1); }

	retval = CVodeSetLinearSolver(cvode_mem, LS, A);
	if (check_retval(&retval, "CVodeSetLinearSolver", 1)) { return (1); }

	
	std::ofstream csv_file("model_results.csv");
	csv_file << "Time,V,Ap,Apm,Thn,The,Tkn,Tke,B,Ps,Pl,Bm,A\n";  // Write the header

	csv_file << t0 << "," << NV_Ith_S(y, 0) << "," << NV_Ith_S(y, 1) << "," << NV_Ith_S(y, 2) << "," << NV_Ith_S(y, 3)
					<< "," << NV_Ith_S(y, 4) << "," << NV_Ith_S(y, 5) << "," << NV_Ith_S(y, 6) << "," << NV_Ith_S(y, 7)
					<< "," << NV_Ith_S(y, 8) << "," << NV_Ith_S(y, 9) << "," << NV_Ith_S(y, 10) << "," << NV_Ith_S(y, 11) << "\n";  // Write data to CSV file

	sunrealtype t = t0;
	sunrealtype dt = 0.1;
	while (t < tf)
	{
		sunrealtype tret;
		retval = CVode(cvode_mem, t + dt, y, &tret, CV_NORMAL);
		
		if (check_retval(&retval, "CVode", 1)) { break; }
		if (retval == CV_SUCCESS)
		{
			t = tret;
			//printf("At time %g: The = %g, Tke = %g, V = %g, A = %g\n", t, NV_Ith_S(y, 4), NV_Ith_S(y, 6), NV_Ith_S(y, 0), NV_Ith_S(y, 11));
			csv_file << t << "," << NV_Ith_S(y, 0) << "," << NV_Ith_S(y, 1) << "," << NV_Ith_S(y, 2) << "," << NV_Ith_S(y, 3)
					<< "," << NV_Ith_S(y, 4) << "," << NV_Ith_S(y, 5) << "," << NV_Ith_S(y, 6) << "," << NV_Ith_S(y, 7)
					<< "," << NV_Ith_S(y, 8) << "," << NV_Ith_S(y, 9) << "," << NV_Ith_S(y, 10) << "," << NV_Ith_S(y, 11) << "\n";  // Write data to CSV file

		}

  	}

	csv_file.close();

	N_VDestroy(y);            
	N_VDestroy(abstol);       
	CVodeFree(&cvode_mem);    
	SUNLinSolFree(LS);        
	SUNMatDestroy(A);         
	SUNContext_Free(&sunctx);

	return 0;
}

// Calcula normal
static real twonorm(vector<real>& v)
{
	real n = 0.0;

	for(int i = 0; i < v.size(); i++)
	{
		n += pow(v[i], 2);
	}

	n = sqrt(n);

	return n;

}

// Subtrai dois vetores
static void subt(vector<real>& v1, vector<real>& v2, vector<real>& vf)
{
	for(int i = 0; i < v1.size(); i++)
	{
		vf.push_back(v1[i] - v2[i]);
	}

	return;
}

// FUNÇÃO OBJETIVO
static real model(real *x)
{
	SUNContext sunctx;
	sunrealtype t0, tf;
	N_Vector y;
	N_Vector abstol;
	SUNMatrix A;
	SUNLinearSolver LS;
	void* cvode_mem;
	int retval;

	y         = NULL;
	abstol    = NULL;
	A         = NULL;
	LS        = NULL;
	cvode_mem = NULL;

	t0 = 0.0;
	tf = 60.0;

	retval = SUNContext_Create(SUN_COMM_NULL, &sunctx);
	if (check_retval(&retval, "SUNContext_Create", 1)) { return (1); }

	y = N_VNew_Serial(NEQ, sunctx);
	if (check_retval((void*)y, "N_VNew_Serial", 0)) { return (1); }

	/*
    Params description:

    x[0]=> V0
    x[1]=> pi_v
    x[2]=> beta_tk
    x[3]=> k_v2
    x[4]=> beta_ap
    x[5]=> delta_apm
    x[6]=> pi_tk
    x[7]=> delta_tk
    x[8]=> delta_a
    x[9]=> alfa_tk
    x[10]=> pi_th
    x[11]=> delta_th
    */
	
	//NV_Ith_S(y, 0) = 2.38303916e+05;
	NV_Ith_S(y, 0) = x[0];
	NV_Ith_S(y, 1) = 1e6;
	NV_Ith_S(y, 2) = 0.0;
	NV_Ith_S(y, 3) = 1e6;
	NV_Ith_S(y, 4) = 0.0;
	NV_Ith_S(y, 5) = 5.0e5;
	NV_Ith_S(y, 6) = 0.0;
	NV_Ith_S(y, 7) = 2.5e5;
	NV_Ith_S(y, 8) = 0.0;
	NV_Ith_S(y, 9) = 0.0;
	NV_Ith_S(y, 10) = 0.0;
	NV_Ith_S(y, 11) = 0.0;

	pi_v = x[1];
    beta_tk = x[2];
    k_v2 = x[3];
    beta_ap = x[4];
    delta_apm = x[5];
    pi_tk = x[6];
    delta_tk = x[7];
    delta_a = x[8];
    alfa_tk = x[9];
    pi_th = x[10];
    delta_th = x[11];
	
	abstol = N_VNew_Serial(NEQ, sunctx);
	if (check_retval((void*)abstol, "N_VNew_Serial", 0)) { return (1); }

	cvode_mem = CVodeCreate(CV_BDF, sunctx);
  	if (check_retval((void*)cvode_mem, "CVodeCreate", 0)) { return (1); }

	retval = CVodeInit(cvode_mem, f, t0, y);
  	if (check_retval(&retval, "CVodeInit", 1)) { return (1); }

	retval = CVodeSStolerances(cvode_mem, RTOL, ATOL);
    if (check_retval(&retval, "CVodeSStolerances", 1)) { return (1); }

	A = SUNDenseMatrix(NEQ, NEQ, sunctx);
	if (check_retval((void*)A, "SUNDenseMatrix", 0)) { return (1); }

	LS = SUNLinSol_Dense(y, A, sunctx);
	if (check_retval((void*)LS, "SUNLinSol_Dense", 0)) { return (1); }

	retval = CVodeSetLinearSolver(cvode_mem, LS, A);
	if (check_retval(&retval, "CVodeSetLinearSolver", 1)) { return (1); }


	vector<real> the_data, tke_data, virus_data, ant_data;

	sunrealtype t = t0;
	sunrealtype dt = 0.1;
	while (t < tf)
	{
		sunrealtype tret;
		retval = CVode(cvode_mem, t + dt, y, &tret, CV_NORMAL);
		
		if (check_retval(&retval, "CVode", 1)) { break; }
		if (retval == CV_SUCCESS)
		{
			t = tret;
			the_data.push_back(NV_Ith_S(y, 4));
			tke_data.push_back(NV_Ith_S(y, 6));
			virus_data.push_back(NV_Ith_S(y, 0));
			ant_data.push_back(NV_Ith_S(y, 11));
			//printf("At time %g: The = %g, Tke = %g, V = %g, A = %g\n", t, NV_Ith_S(y, 4), NV_Ith_S(y, 6), NV_Ith_S(y, 0), NV_Ith_S(y, 11));
		}

  	}

	N_VDestroy(y);            
	N_VDestroy(abstol);       
	CVodeFree(&cvode_mem);    
	SUNLinSolFree(LS);        
	SUNMatDestroy(A);         
	SUNContext_Free(&sunctx);

	if(retval < 0)
	{
		return 1e3;
	}

	
	vector<real> The_e, Tke_e, V_e, A_e;
	//vector<real> The_e, Tke_e;

	The_e = {the_data.begin() + cd4_i, the_data.begin() + cd4_f + 1};
	Tke_e = {tke_data.begin() + cd8_i, tke_data.begin() + cd8_f + 1};
	V_e = {virus_data.begin() + virus_i, virus_data.begin() + virus_f + 1};
	A_e = {ant_data.begin() + ant_i, ant_data.begin() + ant_f + 1};

	real cd4_error, cd8_error, virus_error, ant_error;
	vector<real> cd4_aux, cd8_aux, virus_aux, ant_aux;


	vector<real>::iterator valmin = min_element(V_e.begin(), V_e.end());
	real vmin = *valmin;

	if(vmin <= 0)
	{
		return 1e3;
	}
	else
	{
    	vector<real> varr;

		for(int i = 0; i < V_e.size(); i++)
		{
			varr.push_back(log10(V_e[i]));
		}

		subt(virus_y_v, varr, virus_aux);
		virus_error = twonorm(virus_aux) / twonorm(virus_y_v);

	}


	subt(cd4_y_v, The_e, cd4_aux);
	cd4_error = twonorm(cd4_aux) / twonorm(cd4_y_v);
	
	subt(cd8_y_v, Tke_e, cd8_aux);
	cd8_error = twonorm(cd8_aux) / twonorm(cd8_y_v);

	subt(ant_y_v, A_e, ant_aux);
	ant_error = twonorm(ant_aux) / twonorm(ant_y_v);

	real w1 = 20.0;
	real w2 = 30.0;
	real w3 = 18.0;
	real w4 = 10.0;

	real total = w1 + w2 + w3 + w4;
	//real total = w1 + w2;

	w1 = w1/total;
	w2 = w2/total;
	w3 = w3/total;
	w4 = w4/total;

	real error;
	error = w1*cd4_error + w2*cd8_error + w3*virus_error + w4*ant_error;

	return error;
}

// interpolação linear
static real lagrange(real x, vector<real>& x_v, vector<real>& y_v)
{
	  real x0, x1, y0, y1, L0, L1, P;
	
	  /*
	 	x0 e x1 = valores que cercam x 
	  */
	  for(int i = 0; i < x_v.size(); i++)
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

	  P = L0 * y0 + L1 * y1;

	  return P; // valor interpolado
}

int main()
{
	real t0, tf;
	t0 = 0.0;
	tf = 60.0;
	real step = 0.1;
	vector<real> t;

	for(real v = t0; v < tf; v += step)
	{
		t.push_back(v);
	}

	
	fstream fin, fin1;

    fin.open("CD4_Critical.csv", ios::in);
    string line, word, temp;
	int cont = 0;
	int numdelinhas = 0;

	getline(fin, line);

	while (fin >> temp)
	{

		getline(fin, line);

		//stringstream s(line);
		stringstream s(temp);

		while (cont < 2)
        {
            getline(s, word, ',');

			if (cont == 0)
			{
				cd4_x.push_back(stor(word) + 5.0);
			}
			else
			{
				cd4_y.push_back(stor(word));
			}

            cont += 1;
        }

		getline(fin, line);
		getline(fin, line);
		cont = 0;
		//numdelinhas++;

	}

	fin.close();


    fin.open("CD8_Critical.csv", ios::in);
	cont = 0;

	getline(fin, line);

	while (fin >> temp)
	{

		getline(fin, line);

		//stringstream s(line);
		stringstream s(temp);

		while (cont < 2)
        {
            getline(s, word, ',');

			if (cont == 0)
			{
				cd8_x.push_back(stor(word) + 5.0);
			}
			else
			{
				cd8_y.push_back(stor(word));
			}

            cont += 1;
        }

		getline(fin, line);
		getline(fin, line);
		cont = 0;

	}

	fin.close();

    fin.open("Viral_load.csv", ios::in);
	cont = 0;

	getline(fin, line);

	while (fin >> temp)
	{

		getline(fin, line);

		//stringstream s(line);
		stringstream s(temp);

		while (cont < 2)
        {
            getline(s, word, ',');

			if (cont == 0)
			{
				virus_x.push_back(stor(word) + 5.0);
			}
			else
			{
				virus_y.push_back(stor(word));
			}

            cont += 1;
        }

		getline(fin, line);
		getline(fin, line);
		cont = 0;

	}

	int i = 0;
	while(i < 5)
	{
		virus_x.pop_back();
		virus_y.pop_back();
		i += 1;
	}

	fin.close();

    fin.open("IgG_data.csv", ios::in);
	fin1.open("IgM_data.csv", ios::in);
    string line1, word1, temp1;
	cont = 0;

	getline(fin, line);
	getline(fin1, line1);

	while (fin >> temp)
	{

		fin1 >> temp1;

		getline(fin, line);
		getline(fin1, line1);

		//stringstream s(line);
		stringstream s(temp);
		//stringstream s1(line1);
		stringstream s1(temp1);

		while (cont < 2)
        {
            getline(s, word, ',');
			getline(s1, word1, ',');

			if (cont == 0)
			{
				ant_x.push_back(stor(word) + 5.0);
			}
			else
			{
				ant_y.push_back(stor(word) + stor(word1));
			}

            cont += 1;
        }

		getline(fin, line);
		getline(fin1, line1);
		getline(fin, line);
		getline(fin1, line1);
		cont = 0;

	}

	fin.close();
	fin1.close();


	

	// cd4
	//int cd4_i, cd4_f;
	int tam;
	real a;

	a = round(cd4_x[0] * 10) / 10;
	cd4_i = int((round((a - t0) / step)));
	tam = cd4_x.size() - 1;
	if (cd4_x[tam] > tf)
	{
		cd4_f = t.size() - 1;
	}
	else
	{
		a = int(cd4_x[tam] * 10) / 10.0;
		cd4_f = int((round((a - t0) / step)));
	}

	//vector<real> cd4_y_v((cd4_f - cd4_i + 1), 0.0);
	cd4_y_v = vector<real>((cd4_f - cd4_i + 1), 0.0);
	vector<real> t_cd4;
	real val_x;

	t_cd4 = {t.begin() + cd4_i, t.begin() + cd4_f + 1};

	for(int i = 0; i < cd4_y_v.size(); i++)
	{
		val_x = t_cd4[i];
		cd4_y_v[i] = lagrange(val_x, cd4_x, cd4_y);
	}


	// cd8
	//int cd8_i, cd8_f;

	a = round(cd8_x[0] * 10) / 10;
	cd8_i = int((round((a - t0) / step)));
	tam = cd8_x.size() - 1;
	if (cd8_x[tam] > tf)
	{
		cd8_f = t.size() - 1;
	}
	else
	{
		a = int(cd8_x[tam] * 10) / 10.0;
		cd8_f = int((round((a - t0) / step)));
	}

	//vector<real> cd8_y_v((cd8_f - cd8_i + 1), 0.0);
	cd8_y_v = vector<real>((cd8_f - cd8_i + 1), 0.0);
	vector<real> t_cd8;

	t_cd8 = {t.begin() + cd8_i, t.begin() + cd8_f + 1};

	for(int i = 0; i < cd8_y_v.size(); i++)
	{
		val_x = t_cd8[i];
		cd8_y_v[i] = lagrange(val_x, cd8_x, cd8_y);
	}


	// virus
	//int virus_i, virus_f;

	a = round(virus_x[0] * 10) / 10;
	virus_i = int((round((a - t0) / step)));
	tam = virus_x.size() - 1;
	if (virus_x[tam] > tf)
	{
		virus_f = t.size() - 1;
	}
	else
	{
		a = int(virus_x[tam] * 10) / 10.0;
		virus_f = int((round((a - t0) / step)));
	}

	//vector<real> virus_y_v((virus_f - virus_i + 1), 0.0);
	virus_y_v = vector<real>((virus_f - virus_i + 1), 0.0);
	vector<real> t_virus;

	t_virus = {t.begin() + virus_i, t.begin() + virus_f + 1};

	for(int i = 0; i < virus_y_v.size(); i++)
	{
		val_x = t_virus[i];
		virus_y_v[i] = lagrange(val_x, virus_x, virus_y);
	}


	// anticorpos
	//int ant_i, ant_f;

	a = round(ant_x[0] * 10) / 10;
	ant_i = int((round((a - t0) / step)));
	tam = ant_x.size() - 1;
	if (ant_x[tam] > tf)
	{
		ant_f = t.size() - 1;
	}
	else
	{
		a = int(ant_x[tam] * 10) / 10.0;
		ant_f = int((round((a - t0) / step)));
	}

	//vector<real> ant_y_v((ant_f - ant_i + 1), 0.0);
	ant_y_v = vector<real>((ant_f - ant_i + 1), 0.0);
	vector<real> t_ant;

	t_ant = {t.begin() + ant_i, t.begin() + ant_f + 1};

	for(int i = 0; i < ant_y_v.size(); i++)
	{
		val_x = t_ant[i];
		ant_y_v[i] = lagrange(val_x, ant_x, ant_y);
	}

	
	int n_gens = 1000000;
	int pop_size = 150;
	int num_par = TAM;

	real porcentagem = 0.9;
	real auxMin = 1-porcentagem;
  	real auxMax = 1+porcentagem;

	vector<real> l_bounds = {V0*10, pi_v * 0.00001, beta_tk*0.1, k_v2*1,
							 beta_ap*auxMin, delta_apm*0.01, pi_tk*auxMin,
							 delta_tk*0.01, delta_a*auxMin, alfa_tk*auxMin,
							 pi_th*0.001, delta_th*10};

	vector<real> h_bounds = {V0*100, pi_v * 10, beta_tk*10, k_v2*10,
							 beta_ap*auxMax, delta_apm*1000, pi_tk*auxMax,
							 delta_tk*10000, delta_a*auxMax, alfa_tk*auxMax,
							 pi_th*0.1, delta_th*1000};
	
	printf("n_gens: %d, pop_size: %d, num_par: %d\n", n_gens, pop_size, num_par);
	real *x = differential_evolution(false, model, &l_bounds[0], &h_bounds[0], n_gens, 0.7, 0.5, 1.0, pop_size, num_par, BEST1BIN, time(NULL), 0, 0.01);


	cout << "x: " << endl;
	for(int i = 0; i < num_par; i++)
	{
        	cout << " " << x[i];
    }
	cout << endl;

	printf("\n%f", model(x));

	real func = modelcsv(x);

	std::ofstream csv_file("param_values.csv");
	csv_file << "V0,pi_v,beta_tk,k_v2,beta_ap,delta_apm,pi_tk,delta_tk,delta_a,alfa_tk,pi_th,delta_th\n";  // Write the header

	csv_file << x[0] << "," << x[1] << "," << x[2] << "," << x[3]
			 << "," << x[4] << "," << x[5] << "," << x[6] << "," << x[7]
			 << "," << x[8] << "," << x[9] << "," << x[10] << "," << x[11] << "\n";  // Write data to CSV file

	return 0;
}