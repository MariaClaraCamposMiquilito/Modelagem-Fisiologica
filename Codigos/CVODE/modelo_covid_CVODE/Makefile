# Compilador, opções de compilação.
CXX = mpic++

#CFLAGS = -g -Wall              # use para debug
CFLAGS = -O3 -Wall              # use para compilar com otimização
#CFLAGS = -g -Wall -fopenmp     # use para valgrind
#CFLAGS = -O3 -Wall -fopenmp    # use para OpenMP

# Bibliotecas
LDFLAGS = -lsundials_cvode \
          -lsundials_nvecserial \
          -lsundials_sunmatrixdense \
          -lsundials_sunlinsoldense \
          -lsundials_core

# Fontes
SRCDIR = ./
SRCS   = $(shell ls $(SRCDIR)/*.cpp 2>/dev/null)
DEPS   = $(shell ls $(SRCDIR)/*.h 2>/dev/null)
OBJS   = $(SRCS:.cpp=.o)

# Nome do executável
exec1 = covid_model

# Regras de compilação
all: $(exec1)

$(exec1): $(OBJS) $(DEPS)
	$(CXX) $(CFLAGS) -o $(exec1) $(OBJS) $(LDFLAGS)

clean:
	rm -f $(exec1) $(SRCDIR)/*.o $(SRCDIR)/*.csv