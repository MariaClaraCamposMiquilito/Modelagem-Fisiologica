import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
"""
Scipy é uma biblioteca de computação científica
solve_ivp são funções que resolvem numericamente o sistema de EDOs
    calcula o quanto as populações variam em pequenos passos de tempo
"""

# -----------------------------------------------------------------------------

def modelo(t,y,p): # É chamada pelo solve_ivp
    # 3 argumentos
        # t: o ponto atual no tempo
        # y: array que contém os valores atuais de todas as 15 populações
        # p: "dicionário" contendo todos os parâmetros do modelo

    # Essa parte somente desempacota o vetor y de acordo com as populações
    V   = y[0] # Carga Viral
    Ap  = y[1] # Células APCs naive
    ApM = y[2] # Células APCs maduras/ativadas
    I   = y[3] # Sinal de Inflamação/ Infecção
    ThN = y[4] # Células T helpes naive
    ThE = y[5] # Células T helpes efetoras/ativadas
    TkN = y[6] # Células T killer (citotóxicas naive)
    TkE = y[7] # Células T killer (citotóxicas efetoras)
    B   = y[8] # Células B
    Ps  = y[9] # Plasmócitos de vida curta (IgM)
    Pl  = y[10] # Plasmócitos de vida longa (IgG)
    Bm  = y[11] # Células B de memória
    IgM = y[12] # Anticorpo IgM
    IgG = y[13] # Anticorpo IgG
    C   = y[14] # Citocinas (sinalizadores inflamatórios)
    Nk   = y[15]

    # Parâmetros Basais
    """
    Basal = estado normal, de repouso (valores padrão do corpo, antes de qualquer pertubação - Infecção Viral)
    Representam as condições iniciais , em t=0, a simulação começa com os valores fornecidos por p
    E representam o "Set Point" homeostático -> durante a infecção, o corpo não para de produzir células, ele fica 
    tentando constantemente repor as células para retornar ao seu estado normal
    """
    Ap0  = p['Ap0']
    ThN0 = p['ThN0']
    TkN0 = p['TkN0']
    B0   = p['B0']
    NK0  = p['NK0']
 
    # Equações
    """
    Todas as equações seguem o modelo dX/dt = (Soma dos termos de produção) - (Somas dos termos de remoção/morte)
    """

    # Equação 1 - Vírus
    dVdt   = p['pi_v']*V - p['kv1']*V*IgG - p['kv1']*V*IgM - p['kv2']*V*TkE - p['kv3']*V*ApM 
    
    # Equação 2 - APCs naive
    dApdt  = p['alpha_ap']*(C + 1.0)*(Ap0 - Ap) - p['beta_ap']*Ap*((p['cap1']*V)/(p['cap2'] + V))
    
    # Equação 3 - APCs maduras
    dApMdt = p['beta_ap']*Ap*((p['cap1']*V)/(p['cap2'] + V)) - p['beta_apm']*ApM*V - p['gama_apm']*ApM 
    
    # Equação 4 - Dinâmica das células do SIH
    dIdt   = p['beta_apm']*ApM*V + p['beta_tke']*TkE*V - p['gama_apm']*I - p['gamma_ink']*Nk*I - p['gamma_itk']*TkE*I
    " O que aumenta a população é a infecção das APCs e das TCD8+ pelo vírus"
    " O que reduz a população é um decaimento natural"
    " Adicionei dois termos: a morte das células Infectadas pelas TCD8+ e pelas NK"
    # Equação 5 - Células TCD4+ naive
    dThNdt = p['alpha_th']*(ThN0 - ThN) - p['beta_th']*ApM*ThN
    
    # Equação 6 - Células TCD4+ maduras
    dThEdt = p['beta_th']*ApM*ThN + p['pi_th']*ApM*ThE - p['delta_th']*ThE 
    
    # Equação 7 - Células TCD8+ naive
    dTkNdt = p['alpha_tk']*(C + 1)*(TkN0 - TkN) - p['beta_tk']*(C + 1)*ApM*TkN
    
    # Equação 8 - Células TCD8+ maduras
    dTkEdt = p['beta_tk']*(C + 1)*ApM*TkN + p['pi_tk']*ApM*TkE - p['beta_tke']*TkE*V - p['delta_tk']*TkE
    
    # Equação 9 - Células B
    dBdt   = p['alpha_b']*(B0 - B) + p['pi_b1']*V*B + p['pi_b2']*ThE*B - p['beta_ps']*ApM*B - p['beta_pl']*ThE*B - p['beta_bm']*ThE*B
    
    # Equação 10 - Plasmócitos de vida curta
    dPsdt  = p['beta_ps']*ApM*B - p['delta_ps']*Ps
    
    # Equação 11 - Plasmócitos de vida longa
    dPldt  = p['beta_pl']*ThE*B - p['delta_pl']*Pl + p['delta_bm']*Bm
    
    # Equação 12 - Células B de memória
    dBmdt  = p['beta_bm']*ThE*B + p['pi_bm1']*Bm*(1.0 - (Bm/p['pi_bm2'])) - p['delta_bm']*Bm
    
    # Equação 13 - Anticorpos IgM
    dIgMdt = p['pi_ps']*Ps - p['delta_am']*IgM
    
    # Equação 14 - Anticorpos IgG
    dIgGdt = p['pi_pl']*Pl - p['delta_ag']*IgG
    
    # Equação 15 - Citocinas
    dCdt   = p['pi_capm']*ApM + p['pi_ci']*I + p['pi_ctke']*TkE + p['pi_cnk']*Nk - p['gama_c']*C

    # Equação 16 - Células NK
    dNkdt  = p['qn']*( p['Nmax'] - Nk ) * I - p['dn'] * (Nk - NK0) 

    # A função Modelo retorna um vetor dydt que contém as taxas de variação calculadas para cada população
    # Retorna o que foi calculado nas equações de cada população em relação aquele dado ponto t
    dydt = [dVdt, dApdt, dApMdt, dIdt, dThNdt, 
            dThEdt, dTkNdt, dTkEdt, dBdt, dPsdt,
            dPldt, dBmdt,  dIgMdt,  dIgGdt, dCdt, dNkdt]

    return dydt

# -----------------------------------------------------------------------------


# Script principal main (configura e executa a simulação)
if __name__ == "__main__":
    
    # parametros

    # Declaração de um Dicionário pars vazio
    """
    Por que dicionários e não vetores? Dicionários armazenam dados no formato chave-valor
    Exemplo: 'pi_v' é a chave e 1.47 é o valor associado a ela
    """
    pars = {}
    pars['pi_v']     = 1.47
    pars['kv1']      = 9.82e-3
    pars['kv2']      = 6.10e-5
    pars['kv3']      = 6.45e-2
    pars['alpha_ap'] = 1.0
    pars['beta_ap']  = 1.79e-1
    pars['cap1']     = 8.0
    pars['cap2']     = 8.08e6
    pars['gama_apm'] = 4.0e-2
    pars['beta_apm'] = 1.33e-2
    pars['beta_tke'] = 3.5e-6
    pars['alpha_th'] = 2.17e-4
    pars['beta_th']  = 1.8e-5
    pars['pi_th']    = 1.0e-8
    pars['delta_th']  = 3.0e-1
    pars['alpha_tk'] = 1.0
    pars['beta_tk']  = 1.43e-5
    pars['pi_tk']    = 1.0e-8
    #pars['delta_tk']  = 3.0e-1
    pars['delta_tk']  = 3.0e-2
    #pars['alpha_b']  = 3.58e2
    pars['alpha_b']  = 3.578236584
    pars['pi_b1']    = 8.98e-5
    pars['pi_b2']    = 1.27e-8
    pars['beta_ps']  = 6.0e-6
    pars['beta_pl']  = 5.0e-6
    pars['beta_bm']  = 1.0e-6
    pars['delta_ps']  = 2.5
    pars['delta_pl']  = 3.5e-1
    pars['delta_bm']  = 9.75e-4
    pars['pi_bm1']   = 1.0e-5
    pars['pi_bm2']   = 2.5e3
    pars['pi_ps']    = 8.7e-2
    pars['pi_pl']    = 1.0e-3
    pars['delta_am']  = 7.0e-2
    pars['delta_ag']  = 7.0e-2
    pars['pi_capm']  = 3.28e2
    pars['pi_ci']    = 6.44e-3
    pars['pi_ctke']  = 1.78e-2
    pars['gama_c']   = 7.04e2
    # Parâmetros adicionados
    pars['qn']       = 0.52 # Taxa de 
    pars['dn']       = 0.07 # Taxa de decaimento natural das Natural Killers
    pars['gamma_ink'] =  0.000574 # Taxa de morte das células Infectadas pelas NK
    pars['gamma_itk'] = 0.001 # Taxa de morte das células Infectadas pelas NK
    pars['pi_cnk']    = 0.01 # Taxa de produção de citocinas pelas NK

    # condicoes Iniciais 
    V0   = 61.0  # copies/mL
    Ap0  = 1.0e6 # cells/mL
    ApM0 = 0.0   # cells/mL
    I0   = 0.0   # cells/mL
    ThN0 = 1.0e6 # cells/mL
    ThE0 = 0.0   # cells/mL
    TkN0 = 5.0e5 # cells/mL
    TkE0 = 0.0   # cells/mL
    B0   = 2.5e5 # cells/mL
    Ps0  = 0.0   # cells/mL
    Pl0  = 0.0   # cells/mL
    Bm0  = 0.0   # cells/mL
    IgM0 = 0.0   # S/CO
    IgG0 = 0.0   # S/CO
    C0   = 0.0   # pg/mL
    NK0  = 2.0e5

    # adiciona algumas condicoes iniciais nos parametros
    # As primeiras 4 linhas adicionam os valores basais (Ap0, ThN0...) ao dicionário pars, 
    # para que a função modelo possa acessá-los.
    pars['Ap0'] = Ap0
    pars['ThN0'] = ThN0
    pars['TkN0'] = TkN0
    pars['B0'] = B0
    pars['NK0'] = NK0  
    pars['Nmax'] = 

    # Cria um vetor de estado inicial para ser passado para a função modelo
    y0 = [V0, Ap0, ApM0, I0, ThN0, 
        ThE0, TkN0, TkE0, B0, Ps0, 
        Pl0, Bm0, IgM0, IgG0, C0, NK0]

    # Execução da simulacao
    tf = 35.0  #dias
    dt = 0.01 # Pass de tempo
    N = int(tf/dt)
    t = np.linspace(0,tf,N)

    #sol = odeint(modelo, y0, t, args=(pars,))

    sol = solve_ivp(modelo, [0, tf], y0, args=(pars,), method='Radau')
    # (equações, intervalo de tempo, onde as equações vão começar, passa o dicionário de parâmetros para a função modelo, metódo numérico)
    # O método numérico Radau é um método robusto, bom para executar sistemas stiffs
    t = sol.t

    """
    O resultado sol é um objeto
    t = sol.t -> pega o vetor de tempo que o solver usou para os cálculos

    """
    # sol.y é a matriz onde cada linha é uma série temporal completa de uma variável

    V   = sol.y[0,:] # pega a linha 0 (virus) e todas as colunas (:)
    Ap  = sol.y[1,:]
    ApM = sol.y[2,:]
    I   = sol.y[3,:]
    ThN = sol.y[4,:]
    ThE = sol.y[5,:]
    TkN = sol.y[6,:]
    TkE = sol.y[7,:]
    B   = sol.y[8,:]
    Ps  = sol.y[9,:]
    Pl  = sol.y[10,:]
    Bm  = sol.y[11,:]
    IgM = sol.y[12,:]
    IgG = sol.y[13,:]
    C   = sol.y[14,:]
    NK  = sol.y[15,:]
    # Extrai todas as 15 variáveis -> suas trajetórias ao longo do tempo


    # Visualização Gráfica

    # Viremia
    plt.subplot(2,2,1) # Cria um gráfico 2X2 e seleciona o canto superior esquerdo
    df_viremia = pd.read_csv(r'C:\Users\karla\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\dados_viremia.csv')
    df_viremia.columns = [c.strip() for c in df_viremia.columns] 
    #plt.scatter(df_viremia['x'], df_viremia['y'], color='orange', alpha=0.6, label='Cohort Data')
    plt.plot(t,V, color='red', label='V') # Plota os valores de t no eixo x e de V no eixo Y
    plt.xlabel('Time (days)')
    plt.ylabel('Viremia(copies/mL)')
    plt.legend()
    plt.yscale('log') # Coloca o eixo y em escala logarítimica
    plt.grid() # Adiciona grade no fundo do gráfico

    # Citocinas
    df_dados_carga_viral = pd.read_csv(r'C:\Users\karla\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\dados_carga_viral_8dias.csv')
    df_dados_carga_viral.columns = [c.strip() for c in df_dados_carga_viral.columns] 
    plt.subplot(2,2,2) # Canto superior direito
    #plt.scatter(df_dados_carga_viral['x'], df_dados_carga_viral['y'], color='orange', alpha=0.6, label='Cohort Data')
    plt.plot(t,C, color='blue', label='C')
    plt.xlabel('Time (days)')
    plt.ylabel('Cytokines(pg/mL)')
    plt.legend()
    plt.grid()

    # Anticorpos IgG
    plt.subplot(2,2,3) # Canto inferior esquerdo
    plt.plot(t,IgG, color='orange', label='IgG')
    plt.xlabel('Time (days)')
    plt.ylabel('IgG (S/CO)')
    plt.legend()
    plt.yscale('log')
    plt.ylim(2e-1,2e6)
    plt.grid()

    # Anticorpos IgM
    plt.subplot(2,2,4) # Canto superior direito
    plt.plot(t,IgM, color='black', label='IgM')
    plt.xlabel('Time (days)')
    plt.ylabel('IgM (S/CO)')
    plt.legend()
    plt.yscale('log', base=2)
    plt.ylim(2e-1,2e6)
    plt.grid()

    plt.tight_layout() # Ajusta os gráficos para que os rótulos não se sobreponham.
    plt.show() # Exibe os gráficos

"""
* solve_ivp
- função da biblioteca scipy.integrate
- significa resolver problemas de valor inicial
- Ele é um integrador numérico -> funciona dando pequenos passos de tempo
- Argumentos
1) modelo -> função das regras -> usa a função modelo para saber como o sistema muda
2) intervalo de tempo, onde começa e termina a simulação
3) valores iniciais
4) argumentos adicionais: passa o dicionário pars -> 
    Toda vez que o solver chamar Modelo, vai ser passado pars como terceiro argumento da função modelo
5) método numérico

solve_ivp retorna um objeto (sol) que contém todos os resultados da simulação
sol.t é um vetor com todos os pontos que o solver usou em seus cálculos
sol.y é uma matriz em que cada linha corresponde a uma populaça 

solve_ivp é a ferramenta que pega seu ponto de partida (y0) e suas regras (modelo) e calcula a trajetória 
completa do sistema ao longo do tempo.
"""