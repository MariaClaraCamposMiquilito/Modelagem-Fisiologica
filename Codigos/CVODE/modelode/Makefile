#Compilador, opcoes de compilacao.
CC        = g++

#CFLAGS    = -g -Wall 	 #use para debug
CFLAGS   = -O3 -Wall    #use para compilar com otimização
#CFLAGS   = -g -Wall -fopenmp #use para fazer valgrind
#CFLAGS	= -O3 -Wall -fopenmp #use para compilar otimizado
#CFLAGS  = -O3 -Wall -fopenmp -march=core2 -mfpmath=both #use para compilar otimizado intel
LDFLAGS   = -lm -lm -I/usr/local/include -L/usr/local/lib -lsundials_cvode -lsundials_nvecserial -lsundials_sunmatrixdense -lsundials_sunlinsoldense -lsundials_core -lm -Wl,-rpath,/usr/local/lib

#Fontes 
SRCDIR = ./
SRCS   = $(shell ls $(SRCDIR)/*.cpp 2>/dev/null)
DEPS   = $(shell ls $(SRCDIR)/*.h 2>/dev/null)
OBJS   = $(SRCS:.c=.o)

#Nome do Executavel
exec1  = model_covid19

#Regras de compilacao
all : $(exec1)

$(exec1): $(OBJS) $(DEPS)
	$(CC) $(CFLAGS) -o $(exec1) $(OBJS) $(LDFLAGS)

clean : 
	rm -f $(exec1) $(SRCDIR)/*.o $(SRCDIR)/*.csv
	
