#include <stdio.h> 
#include <stdlib.h>
#include <cvode/cvode.h>             // Solver das EDOs
#include <nvector/nvector_serial.h>  // nvector permire usar vetores Sundials em modo serial -> sem paralelização
#include <sunmatrix/sunmatrix_dense.h> // Access to dense SUNMatrix
#include <sunlinsol/sunlinsol_dense.h> // Access to dense SUNLinearSolver
#include <sundials/sundials_types.h>  // Definir o tipo do SUNDIALS
#include <sundials/sundials_math.h>   // Funções matemáticas
#include <sundials/sundials_context.h> // SUNContext
#include <vector>
#include <fstream>  // Permite criar e escrever arquivos

using realtype = sunrealtype;

using namespace std;

// EQUIVALENTE À FUNÇÃO MODELO NO PYTHON

/*
    * t = tempo atual
    * y = vetor com os valores atuais das variáveis (NVECTOR)
    * ydot = onde armazena as derivadas
    * user_data = serve para passar parâmetros para função
    
    * realtype = tipo de ponto flutuante 
*/

int f(realtype t, N_Vector y, N_Vector ydot, void *user_data) {
    // Parâmetros do modelo
    realtype alpha = 1.1, beta = 0.4, delta = 0.1, gamma = 0.4;

    // NV_Ith_S é um macro para acessar o NVECTOR y
    /*
        NV  -> N_Vector
        Ith -> i-ésimo elemento
        S   -> Serial
        Pegar o i-ésimo elemento de NVECTOR serial
    */

    realtype prey = NV_Ith_S(y, 0);     // y[0]
    realtype predator = NV_Ith_S(y, 1); // y[1]

    NV_Ith_S(ydot, 0) = alpha * prey - beta * prey * predator;
    NV_Ith_S(ydot, 1) = delta * prey * predator - gamma * predator;

    return 0;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    // Criando o contexto do Sundials
    SUNContext sunctx;
    SUNContext_Create(MPI_COMM_WORLD, &sunctx);

    // Vetor de condições iniciais
    // Manda criar um novo vetor serial com espaço para dois elementos 
    N_Vector y = N_VNew_Serial(2, sunctx);

    // Condições iniciais (y0)
    NV_Ith_S(y, 0) = 10.0;  
    NV_Ith_S(y, 1) = 5.0;   

    // Criando o solver
    void *cvode_mem = CVodeCreate(CV_ADAMS, sunctx);
    // CV_ADAMS = método mais indicado para sistemas não rígidos

    // Verificando erros

    // Se o CVODE não for criado corretamente
    if (cvode_mem == NULL) {
        printf("Error: CVodeCreate failed\n");
        return 1;
    }

    realtype t0 = 0.0; // Tempo inicial

    if (CVodeInit(cvode_mem, f, t0, y) != CV_SUCCESS) {
        printf("Error: CVodeInit failed\n");
        return 1;
    }

    // Tolerâncias numéricas
    realtype reltol = 1e-4;
    realtype abstol = 1e-4;


    if (CVodeSStolerances(cvode_mem, reltol, abstol) != CV_SUCCESS) {
        printf("Error: CVodeSStolerances failed\n");
        return 1;
    }

    // Matrix densa 2x2
    SUNMatrix A = SUNDenseMatrix(2, 2, sunctx);
    // Solver linear 
    SUNLinearSolver LS = SUNLinSol_Dense(y, A, sunctx);

    if (CVodeSetLinearSolver(cvode_mem, LS, A) != CV_SUCCESS) {
        printf("Error: CVodeSetLinearSolver failed\n");
        return 1;
    }

    // Vetores para guardar dados -> no formato tempo, valor
    vector<pair<realtype, realtype>> prey_data;
    vector<pair<realtype, realtype>> predator_data;

    // Cria um arquivo chamado "lotka_volterra_results.csv" 
    ofstream csv_file("lotka_volterra_results.csv");
    csv_file << "Time,Prey,Predator\n";  // Write the header

    realtype t = t0;     // Tempo inicial
    realtype t1 = 200.0; // Tempo final
    realtype dt = 0.1;   // Passo de tempo

    while (t < t1) {
        realtype tret;
        if (CVode(cvode_mem, t + dt, y, &tret, CV_NORMAL) != CV_SUCCESS) {
            printf("Error: CVode failed\n");
            return 1;
        }
        t = tret;
        prey_data.push_back(make_pair(t, NV_Ith_S(y, 0)));
        predator_data.push_back(make_pair(t, NV_Ith_S(y, 1)));
        printf("At time %g: Prey = %g, Predator = %g\n", t, NV_Ith_S(y, 0), NV_Ith_S(y, 1));
        csv_file << t << "," << NV_Ith_S(y, 0) << "," << NV_Ith_S(y, 1) << "\n";  // Write data to CSV file
    }

    // Close the CSV file
    csv_file.close();

    N_VDestroy(y);
    SUNLinSolFree(LS);
    SUNMatDestroy(A);
    CVodeFree(&cvode_mem);
    SUNContext_Free(&sunctx);
    MPI_Finalize();

}
