#include "de.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <vector>

using namespace std;

int stop(real *a, int n, real atol, real tol)
{
    real std = stddev(a, n);
    real aux = atol + tol * abs(mean(a, n));
    if(std <= aux)
    {
        return 0;
    }
    return 1;
}

real stddev(real *a, int n)
{
    real fsum = 0.0, m = mean(a, n);
    for(int i = 0; i < n; i++)
    {
        fsum += (a[i] - m) * (a[i] - m);
    }
    return sqrt(fsum / n);
}

real mean(real *a, int n)
{
    real sum = 0.0;
    for(int i = 0; i < n; i++)
    {
        sum += a[i];
    }
    return sum / n;
}

real **load_pop(const char *filename, int pop_size, int num_par) {
    real **filedata = (real**)malloc(sizeof(real*) * pop_size);
    for (int i = 0; i < pop_size; i++) {
        filedata[i] = (real*)malloc(sizeof(real) * num_par);
    }

    FILE *fp = fopen(filename, "r");
    if(fp == NULL) {
        fprintf(stderr, "Error reading file %s\n", filename);
        return NULL;
    }

    real data;
    char ch;
    for (int i = 0; i < pop_size; i++) {
        for(int j = 0; j < num_par; j++) {
            fscanf(fp, "%lf", &data);
            filedata[i][j] = data;
        }
        while((ch = fgetc(fp)) != '\n' && ch != EOF);
    }
    fclose(fp);
    return filedata;
}

static real rnd_uni(long *idum) {
    long j;
    long k;
    static long idum2 = 123456789;
    static long iy = 0;
    static long iv[NTAB];
    real temp;

    if(*idum <= 0) {
        if(-(*idum) < 1)
            *idum = 1;
        else
            *idum = -(*idum);
        idum2 = (*idum);
        for(j = NTAB + 7; j >= 0; j--) {
            k = (*idum) / IQ1;
            *idum = IA1 * (*idum - k * IQ1) - k * IR1;
            if(*idum < 0)
                *idum += IM1;
            if(j < NTAB)
                iv[j] = *idum;
        }
        iy = iv[0];
    }
    k = (*idum) / IQ1;
    *idum = IA1 * (*idum - k * IQ1) - k * IR1;
    if(*idum < 0)
        *idum += IM1;
    k = idum2 / IQ2;
    idum2 = IA2 * (idum2 - k * IQ2) - k * IR2;
    if(idum2 < 0)
        idum2 += IM2;
    j = iy / NDIV;
    iy = iv[j] - idum2;
    iv[j] = *idum;
    if(iy < 1)
        iy += IMM1;
    if((temp = AM * iy) > RNMX)
        return RNMX;
    else
        return temp;
}

real *differential_evolution(bool from_file, evalfn *evaluate, real *inibound_l, real *inibound_h, int genmax, real CR,
                             real F_min, real F_max, int NP, int D, int strategy, long seed, real atol, real tol, MPI_Comm comm) {

    int rank, num_procs;
    // cada processo tem um rank, o processo de rank 0 é o mestre
    // num_procs = quantidade total de processos na execução de MPI (valor que a gente passa na flag)
    // MPI_Comm -> comunicador que conecta todos os processos envolvidos no DE
    MPI_Comm_rank(comm, &rank);
    // mpi retorna qual o identificador da instancia atual e armazaena em rank
    MPI_Comm_size(comm, &num_procs);
    // descobre quantos processos foram requisitados no mpirun e armazena em num_procs
    
    // Divisão de carga do NP entre os ranks
    int q = NP / num_procs; // quantidade de indivíduos que todo processo precisa avaliar
    int rem = NP % num_procs; // representa os indivíduos que sobraram, vao ser redistribuídos 

    vector<int> counts(num_procs); // quantos indivíduos o processo de rank vai processar
    vector<int> displs(num_procs); // indica em qual posição do vetor global de indivíduos começam os dados do processo
    vector<int> counts_params(num_procs); // quantidade real de valores numéricos que serão transmitidos para cada rank
    vector<int> displs_params(num_procs);

    // Percorre todos os processos de $0$ a $P-1$ e preenche as cotas: os primeiros rem processos pegam $q + 1$ indivíduos, e os restantes pegam $q$.
    for (int r = 0; r < num_procs; r++) {
        counts[r] = (r < rem) ? (q + 1) : q;
        displs[r] = (r < rem) ? (r * (q + 1)) : (r * q + rem);
        counts_params[r] = counts[r] * D;
        displs_params[r] = displs[r] * D;
    }

    int local_n = counts[rank]; 
    // int local_n = counts[rank];: Guarda na variável local quantos indivíduos cabem exclusivamente ao processo atual calcular.

    real F = F_min;
    long rnd_uni_init = -(long)seed;

    // Alocação de Memória Contígua e Buffers
    real *cost = (real*)malloc(sizeof(real) * NP);
    real *trial_cost = (real*)malloc(sizeof(real) * NP);
    real *local_cost = (real*)malloc(sizeof(real) * local_n);

    real *best = (real*)malloc(sizeof(real) * D);
    real *bestit = (real*)malloc(sizeof(real) * D);
    real *tmp = (real*)malloc(sizeof(real) * D);
    real *original_ind = (real*)malloc(sizeof(real) * D);

    // Buffers contíguos para o Scatterv
    real *trial_pop = (real*)malloc(sizeof(real) * NP * D);
    real *local_trial_pop = (real*)malloc(sizeof(real) * local_n * D);

    real **c = (real **)malloc(sizeof(real*) * NP);
    real **d = (real **)malloc(sizeof(real*) * NP);
    for(int i = 0; i < NP; i++) {
        d[i] = (real*)malloc(sizeof(real) * D);
    }

    /*------ Inicialização da População ------*/
    if(from_file) {
        c = load_pop("population.txt", NP, D);
    } else {
        for(int i = 0; i < NP; i++) {
            c[i] = (real*)malloc(sizeof(real) * D);
        }
    }

    for(int i = 0; i < NP; i++) {
        for(int j = 0; j < D; j++) {
            if(from_file) {
                c[i][j] = inibound_l[j] + c[i][j] * (inibound_h[j] - inibound_l[j]);
            } else {
                c[i][j] = inibound_l[j] + rnd_uni(&rnd_uni_init) * (inibound_h[j] - inibound_l[j]);
            }
            trial_pop[i * D + j] = c[i][j]; // Empacota população inicial para avaliar em paralelo
        }
    }

    // Distribui os indivíduos iniciais
    MPI_Scatterv(trial_pop, counts_params.data(), displs_params.data(), MPI_DOUBLE,
                 local_trial_pop, local_n * D, MPI_DOUBLE, 0, comm);

    // Avaliação paralela inicial das EDOs
    for(int i = 0; i < local_n; i++) {
        local_cost[i] = evaluate(&local_trial_pop[i * D]);
    }

    // Coleta custos iniciais para todos os ranks
    MPI_Allgatherv(local_cost, local_n, MPI_DOUBLE,
                   cost, counts.data(), displs.data(), MPI_DOUBLE, comm);

    real cmin = cost[0];
    int imin = 0;
    for(int i = 1; i < NP; i++) {
        if(cost[i] < cmin) {
            cmin = cost[i];
            imin = i;
        }
    }

    memcpy(best, c[imin], D * sizeof(real));
    memcpy(bestit, c[imin], D * sizeof(real));

    real ***pold = &c;
    real ***pnew = &d;
    real ***pswap;

    int r1, r2, r3, r4, r5;
    int gen = 0;
    int n, L;

    /*=======================================================================*/
    /*========= Iteration loop ==============================================*/
    /*=======================================================================*/

    while((gen < genmax) && (stop(cost, NP, atol, tol))) {
        gen++;

        // 1. Gera todos os indivíduos de teste da geração no Rank 0
        if(rank == 0) {
            for(int i = 0; i < NP; i++) {
                do { r1 = (int)(rnd_uni(&rnd_uni_init) * NP); } while(r1 == i);
                do { r2 = (int)(rnd_uni(&rnd_uni_init) * NP); } while((r2 == i) || (r2 == r1));
                do { r3 = (int)(rnd_uni(&rnd_uni_init) * NP); } while((r3 == i) || (r3 == r1) || (r3 == r2));
                do { r4 = (int)(rnd_uni(&rnd_uni_init) * NP); } while((r4 == i) || (r4 == r1) || (r4 == r2) || (r4 == r3));
                do { r5 = (int)(rnd_uni(&rnd_uni_init) * NP); } while((r5 == i) || (r5 == r1) || (r5 == r2) || (r5 == r3) || (r5 == r4));

                if(strategy == BEST1BIN) {
                    memcpy(tmp, (*pold)[i], D * sizeof(real));
                    n = (int)(rnd_uni(&rnd_uni_init) * D);
                    real F_J = F_min + rnd_uni(&rnd_uni_init) * (F_max - F_min);
                    for(L = 0; L < D; L++) {
                        if((rnd_uni(&rnd_uni_init) < CR) || L == (D - 1)) {
                            tmp[n] = bestit[n] + F_J * ((*pold)[r2][n] - (*pold)[r3][n]);
                        }
                        n = (n + 1) % D;
                    }
                }
                else if(strategy == RAND1BIN) {
                    memcpy(tmp, (*pold)[i], D * sizeof(real));
                    n = (int)(rnd_uni(&rnd_uni_init) * D);
                    for(L = 0; L < D; L++) {
                        if((rnd_uni(&rnd_uni_init) < CR) || L == (D - 1)) {
                            tmp[n] = (*pold)[r1][n] + F * ((*pold)[r2][n] - (*pold)[r3][n]);
                        }
                        n = (n + 1) % D;
                    }
                }
                else { // RAND2BIN fallback
                    memcpy(tmp, (*pold)[i], D * sizeof(real));
                    n = (int)(rnd_uni(&rnd_uni_init) * D);
                    for(L = 0; L < D; L++) {
                        if((rnd_uni(&rnd_uni_init) < CR) || L == (D - 1)) {
                            tmp[n] = (*pold)[r5][n] + ((*pold)[r1][n] + (*pold)[r2][n] - (*pold)[r3][n] - (*pold)[r4][n]) * F;
                        }
                        n = (n + 1) % D;
                    }
                }

                // Bounce-back
                for(int j = 0; j < D; j++) {
                    if(tmp[j] < inibound_l[j]) {
                        tmp[j] = inibound_l[j] + rnd_uni(&rnd_uni_init) * ((*pold)[i][j] - inibound_l[j]);
                    }
                    if(tmp[j] > inibound_h[j]) {
                        tmp[j] = inibound_h[j] + rnd_uni(&rnd_uni_init) * ((*pold)[i][j] - inibound_h[j]);
                    }
                    trial_pop[i * D + j] = tmp[j];
                }
            }
        }

        // 2. Broadcast de trial_pop para todos os ranks conseguirem atualizar pnew depois
        MPI_Bcast(trial_pop, NP * D, MPI_DOUBLE, 0, comm);

        // 3. Distribui os indivíduos entre os processos para integração das EDOs
        MPI_Scatterv(trial_pop, counts_params.data(), displs_params.data(), MPI_DOUBLE,
                     local_trial_pop, local_n * D, MPI_DOUBLE, 0, comm);

        // 4. Integração das EDOs (gargalo computacional paralelizado)
        for(int i = 0; i < local_n; i++) {
            local_cost[i] = evaluate(&local_trial_pop[i * D]);
        }

        // 5. Coleta todos os custos calculados para todos os nós
        MPI_Allgatherv(local_cost, local_n, MPI_DOUBLE,
                       trial_cost, counts.data(), displs.data(), MPI_DOUBLE, comm);

        // 6. Seleção síncrona em todos os ranks
        for(int i = 0; i < NP; i++) {
            if(trial_cost[i] <= cost[i]) {
                cost[i] = trial_cost[i];
                memcpy((*pnew)[i], &trial_pop[i * D], D * sizeof(real));
                if(trial_cost[i] < cmin) {
                    cmin = trial_cost[i];
                    memcpy(best, &trial_pop[i * D], D * sizeof(real));
                }
            } else {
                memcpy((*pnew)[i], (*pold)[i], D * sizeof(real));
            }
        }

        memcpy(bestit, best, D * sizeof(real));

        if(rank == 0) {
            cout << "Geracao " << gen << " | Melhor erro = " << cmin << endl;
        }

        pswap = pold;
        pold = pnew;
        pnew = pswap;
    }

    if(rank == 0) {
        printf("\tit: %d\n", gen);
    }

    // Liberação de memória auxiliar
    free(cost);
    free(trial_cost);
    free(local_cost);
    free(bestit);
    free(tmp);
    free(original_ind);
    free(trial_pop);
    free(local_trial_pop);
    for(int i = 0; i < NP; i++) {
        free(d[i]);
    }
    free(d);

    return best;
}