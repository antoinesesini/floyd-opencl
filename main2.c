#include <OpenCL/opencl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>


// Fonction pour initialiser le graphe de test (exemple donné dans l'énnoncé)

void initialiserGraphe(int **graphe, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                graphe[i][j] = 0; // Arc de i à i
            } else if (i < n - 1 && j == i + 1) {
                graphe[i][j] = 2; // Arc de i à i+1 avec un poids de 2
            } else if (i == n - 1 && j == 0) {
                graphe[i][j] = 5; // Arc de n-1 à 0 avec un poids de 5
            } else {
                graphe[i][j] = 5 * n; // Autres arcs
            }
        }
    }
}

/*
void initialiserGraphe(int **graphe, int n) {
   graphe[0][0] = 0;
   graphe[0][1] = INT_MAX;
   graphe[0][2] = -2;
   graphe[0][3] = INT_MAX;
   graphe[1][0] = 4;
   graphe[1][1] = 0;
   graphe[1][2] = 3;
   graphe[1][3] = INT_MAX;
   graphe[2][0] = INT_MAX;
   graphe[2][1] = INT_MAX;
   graphe[2][2] = 0;
   graphe[2][3] = 2;
   graphe[3][0] = INT_MAX;
   graphe[3][1] = -1;
   graphe[3][2] = INT_MAX;
   graphe[3][3] = 0;
}
*/


// Fonction pour lire le fichier contenant le kernel OpenCL
char* load_program_source(const char *filename) {
    FILE *fp;
    char *source;
    int sz=0;
    struct stat status;

    fp = fopen(filename, "r");
    if (fp == 0){
        fprintf(stderr, "Echec lors de l'ouverture\n");
        exit(1);
    }

    if (stat(filename, &status) == 0) // Stocker dans status les informations de filename
        sz = (int) status.st_size; // Récupère la taille en octet de filename

    source = (char *) malloc(sz + 1);
    fread(source, sz, 1, fp); // Lit le contenu de filename et le copie dans le tableau source
    source[sz] = '\0';

    return source;
}

// Fonction de l'algorithme de Floyd
void floyd_opencl(int* graph, int n) {
    char* programSource = load_program_source("floyd2.cl");

    size_t datasize = sizeof(int) * n * n;
    
    cl_int status;

    // STEP 1: Discover and initialize the platforms
    cl_uint numPlatforms = 0;
    cl_platform_id *platforms = NULL;

    // Calcul du nombre de plateformes
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    printf("Number of platforms = %d\n", numPlatforms);

    // Allocation de l'espace
    platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));

    // Trouver les plateformes
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);

    char Name[1000];
    clGetPlatformInfo(platforms[0], CL_PLATFORM_NAME, sizeof(Name), Name, NULL);
    printf("Name of platform : %s\n", Name);
    fflush(stdout);

    // STEP 2: Discover and initialize the devices
    cl_uint numDevices;
    cl_device_id *devices = NULL;

    // Calcul du nombre de périphériques
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
    printf("Number of devices = %d\n", (int)numDevices);

    // Allocation de l'espace
    devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));

    // Trouver les périphériques
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);

    for (cl_uint i = 0; i < numDevices; i++) {
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(Name), Name, NULL);
        printf("Name of device %d: %s\n\n", i, Name);
    }

    // STEP 3: Create a context
    printf("Création du contexte\n");
    fflush(stdout);

    cl_context context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);

    // STEP 4: Create a command queue
    printf("Création de la file d'attente\n");
    fflush(stdout);

    cl_command_queue cmdQueue = clCreateCommandQueue(context, devices[0], 0, &status);

    // STEP 5: Create device buffers
    printf("Création des buffers\n");
    fflush(stdout);
    cl_mem bufferGraph = clCreateBuffer(context, CL_MEM_READ_WRITE, datasize, NULL, &status);

    // STEP 6: Write host data to device buffers
    printf("Ecriture dans les buffers\n");
    fflush(stdout);
    status = clEnqueueWriteBuffer(cmdQueue, bufferGraph, CL_TRUE, 0, datasize, graph, 0, NULL, NULL);

    // STEP 7: Create and compile the program
    printf("CreateProgramWithSource\n");
    fflush(stdout);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&programSource, NULL, &status);
    free(programSource);
    printf("Compilation\n");
    fflush(stdout);

    status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);
    if (status) printf("ERREUR A LA COMPILATION: %d\n", status);

    // STEP 8: Create the kernel
    printf("Création du kernel\n");
    fflush(stdout);
    cl_kernel kernel = clCreateKernel(program, "floyd_step", &status);

    // STEP 9: Set the kernel arguments
    printf("Passage des paramètres\n");
    fflush(stdout);
    size_t globalWorkSize[2] = { n, n }; // Espace global : une grille 2D (n x n)
    for (int k = 0; k < n; k++) {
        //clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferGraph);
        //clSetKernelArg(kernel, 1, sizeof(int), &n);
        clSetKernelArg(kernel, 2, sizeof(int), &k);
        status = clEnqueueNDRangeKernel(cmdQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL); // Exécution du kernel
        clFinish(cmdQueue);
    }
    status = clEnqueueReadBuffer(cmdQueue, bufferGraph, CL_TRUE, 0, datasize, graph, 0, NULL, NULL); // Récupération des résultats

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(bufferGraph);
    clReleaseCommandQueue(cmdQueue);
    clReleaseContext(context);
    free(platforms);
    free(devices);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage : %s <nombre_de_noeuds>\n", argv[0]);
        return 1;
    }

    // Lecture du nombre de nœuds et allocation dynamique de la matrice d'adjacence
    int n = atoi(argv[1]);
    if (n <= 0) {
        fprintf(stderr, "Erreur : le nombre de nœuds doit être un entier positif.\n");
        return 1;
    }
    int** graphe = (int**)malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++) {
        graphe[i] = (int*)malloc(n * sizeof(int));
    }

    // Initialisation du graphe (exemple de graphe de test donné dans le sujet)
    initialiserGraphe(graphe, n);

    // Conversion de la matrice 2D en 1D pour la passer à OpenCL
    int* graph = (int*)malloc(sizeof(int) * n * n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            graph[i * n + j] = graphe[i][j];
        }
    }

    // Affichage de la matrice initiale
    printf("Matrice initiale :\n");
    for (int i = n-1; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (graphe[i][j] == INT_MAX) {
                printf("INF ");
            } else {
                printf("%d ", graphe[i][j]);
            }
        }
        printf("\n");
    }

    printf("\n\n----------------------------------------\n\n");

    floyd_opencl(graph, n); // Appeler l'algorithme de Floyd sur le graphe initialisé en 1D
    
    // Affichage de la matrice des plus courts chemins résultantes
    printf("\nMatrice des plus courts chemins :\n");
    for (int i = n-1; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", graph[i * n + j]);
        }
        printf("\n");
    }

    for (int i = 0; i < n; i++) {
        free(graphe[i]);
    }
    free(graphe);
    free(graph);

    return 0;
}