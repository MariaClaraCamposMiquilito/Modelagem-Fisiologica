# GRÁFICOS

## Viremia
plt.plot(sol.t, np.log10(sol.y[0] + 1), color='red')
plt.title("Vírus")
plt.xlabel("t (dias)")
plt.ylabel("V")
plt.grid()

dataset_viremia = pd.read_csv(r"C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\covid19_model\data\Viral_load.csv", sep = ',')

x = dataset_viremia[dataset_viremia.type == 'mean']['x'] + 5
y = np.log10(dataset_viremia[dataset_viremia.type == 'mean']['y'] + 1)

dataset_viremia_up = np.log10(dataset_viremia[dataset_viremia.type == 'up']['y'] + 1)
dataset_viremia_down = np.log10(dataset_viremia[dataset_viremia.type == 'down']['y'] + 1)
dataset_viremia_mean = np.log10(dataset_viremia[dataset_viremia.type == 'mean']['y'] + 1)

y_error = [dataset_viremia_mean.to_numpy() - dataset_viremia_down.to_numpy(), dataset_viremia_up.to_numpy() - dataset_viremia_mean.to_numpy()]

plt.errorbar(x,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='red', capsize=4, elinewidth=1)

plt.show()

## Citocinas
plt.plot(sol.t, sol.y[14], color='blue')
plt.title("Citocinas")
plt.xlabel("t (dias)")
plt.ylabel("C")
plt.grid()

dataset_il6 = pd.read_csv(r"C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\covid19_model\data\dataset_il6_survivor.csv", sep = ',')

x = dataset_il6[dataset_il6.type == 'mean']['x']
y = dataset_il6[dataset_il6.type == 'mean']['y']

dataset_il6_up = dataset_il6[dataset_il6.type == 'up']['y']
dataset_il6_down = dataset_il6[dataset_il6.type == 'down']['y']
dataset_il6_mean = dataset_il6[dataset_il6.type == 'mean']['y']

y_error = [dataset_il6_mean.to_numpy() - dataset_il6_down.to_numpy(), dataset_il6_up.to_numpy() - dataset_il6_mean.to_numpy()]

plt.errorbar(x,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='blue', capsize=4, elinewidth=1)

plt.show()

## IgG
plt.plot(sol.t, np.log2(sol.y[13] + 1), color='brown')
plt.title("Anticorpos IgG")
plt.xlabel("t (dias)")
plt.ylabel("$I_{gG}$")
plt.grid()

dataset_IgG = pd.read_csv(r"C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\covid19_model\data\IgG_data.csv", sep = ',')

x = dataset_IgG[dataset_IgG.type == 'mean']['x']
y = np.log2(dataset_IgG[dataset_IgG.type == 'mean']['y'] + 1)

dataset_IgG_up = np.log2(dataset_IgG[dataset_IgG.type == 'up']['y'] + 1)
dataset_IgG_down = np.log2(dataset_IgG[dataset_IgG.type == 'down']['y'] + 1)
dataset_IgG_mean = np.log2(dataset_IgG[dataset_IgG.type == 'mean']['y'] + 1)

y_error = [dataset_IgG_mean.to_numpy() - dataset_IgG_down.to_numpy(), dataset_IgG_up.to_numpy() - dataset_IgG_mean.to_numpy()]

plt.errorbar(x,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='brown', capsize=4, elinewidth=1)

plt.show()

# IgM
plt.plot(sol.t, np.log2(sol.y[12] + 1), color='orange')
plt.title("Anticorpos IgM")
plt.xlabel("t (dias)")
plt.ylabel("$I_{gM}$")
plt.grid()

dataset_IgM = pd.read_csv(r"C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\covid19_model\data\IgM_data.csv", sep = ',')

x = dataset_IgM[dataset_IgM.type == 'mean']['x']
y = np.log2(dataset_IgM[dataset_IgM.type == 'mean']['y'] + 1)

dataset_IgM_up = np.log2(dataset_IgM[dataset_IgM.type == 'up']['y'] + 1)
dataset_IgM_down = np.log2(dataset_IgM[dataset_IgM.type == 'down']['y'] + 1)
dataset_IgM_mean = np.log2(dataset_IgM[dataset_IgM.type == 'mean']['y'] + 1)

y_error = [dataset_IgM_mean.to_numpy() - dataset_IgM_down.to_numpy(), dataset_IgM_up.to_numpy() - dataset_IgM_mean.to_numpy()]

plt.errorbar(x,y, yerr = y_error, linestyle='None', label='Data', fmt='o', color='orange', capsize=4, elinewidth=1)

plt.show()

## Natural Killers
dataset_NK = pd.read_csv(r"C:\Users\mique\OneDrive\Documentos\UFJF\Modelagem Fisiologica\Codigos\meus_dados\NK_covid_severo.csv", sep = ',')

plt.scatter(dataset_NK['x'].values, np.log10(dataset_NK['y']), label = 'Dados', color = 'green')
plt.grid()
plt.plot(sol.t, np.log10(sol.y[15] + 1), color = 'green')
plt.title("Células Natural Killers")
plt.xlabel("t (dias)")
plt.ylabel("$NK$")
plt.show()
#plt.plot(sol.t, np.log10)

