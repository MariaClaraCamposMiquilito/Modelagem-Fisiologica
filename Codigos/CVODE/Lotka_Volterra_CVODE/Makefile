# Compilador, opcoes de compilacao.
CXX        = mpic++

#CFLAGS    = -g -Wall 	 # use para debug
CFLAGS   = -O3 -Wall    # use para compilar com otimização
#CFLAGS   = -g -Wall -fopenmp #use para fazer valgrind
#CFLAGS	= -O3 -Wall -fopenmp #use para compilar otimizado
#CFLAGS  = -O3 -Wall -fopenmp -march=core2 -mfpmath=both #use para compilar otimizado intel
LDFLAGS   = -lsundials_cvode -lsundials_nvecserial -lsundials_sunmatrixdense -lsundials_sunlinsoldense -lsundials_core

#-lm                          → biblioteca matemática
#-I/usr/local/include         → onde procurar arquivos .h
#-L/usr/local/lib             → onde procurar bibliotecas
#-lsundials_cvode             → biblioteca do CVODE
#-lsundials_nvecserial        → vetor serial do SUNDIALS
#-lsundials_sunmatrixdense    → matriz densa
#-lsundials_sunlinsoldense    → solver linear denso

# Fontes 
SRCDIR = ./
SRCS   = $(shell ls $(SRCDIR)/*.cpp 2>/dev/null)
DEPS   = $(shell ls $(SRCDIR)/*.h 2>/dev/null)
OBJS   = $(SRCS:.cpp=.o)

# Nome do Executavel
exec1  = lotka_volterra

# Regras de compilacao
all : $(exec1)

$(exec1): $(OBJS) $(DEPS)
	$(CXX) $(CFLAGS) -o $(exec1) $(OBJS) $(LDFLAGS)
clean : 
	rm -f $(exec1) $(SRCDIR)/*.o $(SRCDIR)/*.csv
	
