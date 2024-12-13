__kernel void floyd_step(
    __global int* graph, // Matrice d'adjacence représentée comme un tableau 1D
    const int n,         // Nombre de nœuds dans le graphe
    const int k          // Nœud intermédiaire actuel
) {
    // Récupérer les indices globaux des work-items
    int i = get_global_id(0); // Ligne actuelle
    int j = get_global_id(1); // Colonne actuelle

    // Calcul des indices 1D pour accéder aux éléments de la matrice
    int idx_ij = i * n + j;   // Élement [i][j]
    int idx_ik = i * n + k;   // Élement [i][k]
    int idx_kj = k * n + j;   // Élement [k][j]

    #define INT_MAX 2147483647

    // Met à jour le plus court chemin via le nœud intermédiaire k
    if (graph[idx_ik] != INT_MAX && graph[idx_kj] != INT_MAX) {
        if (graph[idx_ij] > graph[idx_ik] + graph[idx_kj]) {
            graph[idx_ij] = graph[idx_ik] + graph[idx_kj];
        }
    }
}