import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score
import random
import copy

class Node:
    def __init__(self, node_id):
        self.id = node_id
        self.cluster = None

class Cluster:
    def __init__(self, nodes=None):
        self.nodes = set(nodes) if nodes else set()
        self._cached_value = None  # 캐시 추가
        self._cached_key = None

    def add_node(self, node):
        self.nodes.add(node)
        node.cluster = self
        self._cached_key = None  # 캐시 무효화

    def remove_node(self, node):
        self.nodes.discard(node)
        if node.cluster == self:
            node.cluster = None
        self._cached_key = None  # 캐시 무효화

    def internal_edge_sum(self, A):
        ids = np.array([node.id for node in self.nodes])
        subgraph = A[np.ix_(ids, ids)]
        return np.sum(subgraph) - np.trace(subgraph)

    def external_edge_sum(self, A, all_nodes):
        cluster_ids = np.array([n.id for n in self.nodes])
        other_ids = np.array([n.id for n in all_nodes if n.cluster != self])
        if len(cluster_ids) == 0 or len(other_ids) == 0:
            return 0.0
        subgraph = A[np.ix_(cluster_ids, other_ids)]
        return np.sum(subgraph)

    def value(self, A, all_nodes):
        if len(self.nodes) < 2:
            return 0.0

        key = tuple(sorted(n.id for n in self.nodes))
        if self._cached_key == key and self._cached_value is not None:
            return self._cached_value

        internal = self.internal_edge_sum(A)
        external = self.external_edge_sum(A, all_nodes)
        size = len(self.nodes)
        density = internal / (size * (size - 1) + 1e-6)
        contrast = internal / (external + 1e-6)
        base = contrast + density
        penalty = 0.2 * np.sqrt(size)
        self._cached_value = base - penalty
        self._cached_key = key
        return self._cached_value

def softmax(payoffs, beta):
    payoffs = np.array(payoffs)
    scaled = beta * payoffs
    scaled -= np.max(scaled)  # overflow 방지
    exps = np.exp(scaled)
    return exps / np.sum(exps)

def merge_clusters(clusters, A, threshold=0.8):
    merged = []
    used = set()
    for i, ci in enumerate(clusters):
        if i in used:
            continue
        for j in range(i + 1, len(clusters)):
            if j in used:
                continue
            cj = clusters[j]
            links = sum(A[n1.id][n2.id] for n1 in ci.nodes for n2 in cj.nodes)
            possible = len(ci.nodes) * len(cj.nodes)
            if possible == 0:
                continue
            connectivity = links / possible
            if connectivity > threshold:
                ci.nodes.update(cj.nodes)
                for n in cj.nodes:
                    n.cluster = ci
                used.add(j)
        merged.append(ci)
    return merged

def extract_cluster_labels(clusters, num_nodes):
    labels = np.zeros(num_nodes, dtype=int)
    for i, cluster in enumerate(clusters):
        for node in cluster.nodes:
            labels[node.id] = i
    return labels

def initialize_nodes_and_clusters(A):
    nodes = [Node(i) for i in range(len(A))]
    clusters = [Cluster([node]) for node in nodes]
    for node, cluster in zip(nodes, clusters):
        node.cluster = cluster
    return nodes, clusters

def refine_clusters(A, nodes, clusters, beta=1.0, lambda_base=0.1, lambda_growth=0.01, lambda_decay=0.05, max_iter=100, convergence_threshold=1e-4):
    for t in range(max_iter):
        mu_values = []
        for node in nodes:
            old_cluster = node.cluster
            if old_cluster:
                old_cluster.remove_node(node)

            payoffs = []
            cluster_candidates = clusters[:]

            for cluster in cluster_candidates:
                cluster.add_node(node)
                v_with = cluster.value(A, nodes)
                cluster.remove_node(node)
                v_without = cluster.value(A, nodes)
                u = v_with - v_without
                payoffs.append(u)
                mu_values.append(u)

            singleton = Cluster([node])
            u_new = singleton.value(A, nodes)
            mu_values.append(u_new)

            lambda_t = lambda_base + lambda_growth * t + lambda_decay * len(clusters)
            payoffs.append(u_new - lambda_t)

            probs = softmax(payoffs, beta)
            choice = np.random.choice(len(payoffs), p=probs)

            if choice == len(cluster_candidates):
                clusters.append(singleton)
            else:
                cluster_candidates[choice].add_node(node)

        clusters = [c for c in clusters if len(c.nodes) > 0]
        clusters = merge_clusters(clusters, A, threshold=0.5)
        mu = np.mean(mu_values)
        # print(f"Iteration {t}: Avg payoff delta = {mu:.4f}, Num clusters = {len(clusters)}")
        if abs(mu) < convergence_threshold:
            break
    return clusters

def clustering_algorithm(A, beta=3.0, lambda_base=0.1, lambda_growth=0.01, lambda_decay=0.05, max_iter=100, convergence_threshold=1e-4):
    nodes, clusters = initialize_nodes_and_clusters(A)
    return refine_clusters(A, nodes, clusters, beta, lambda_base, lambda_growth, lambda_decay, max_iter, convergence_threshold)

def add_node_and_resettle(A, clusters, beta, lambda_base, lambda_growth, lambda_decay, current_iter, connection_row, K=5):
    n = len(A)
    new_id = n
    A = np.pad(A, ((0, 1), (0, 1)), mode='constant')
    A[new_id, :-1] = connection_row
    A[:-1, new_id] = connection_row

    v_new = Node(new_id)

    payoffs = []
    for c in clusters:
        c.add_node(v_new)
        v_with = c.value(A, [node for cl in clusters for node in cl.nodes] + [v_new])
        c.remove_node(v_new)
        v_without = c.value(A, [node for cl in clusters for node in cl.nodes])
        payoffs.append(v_with - v_without)

    singleton = Cluster([v_new])
    lambda_t = lambda_base + lambda_growth * current_iter + lambda_decay * len(clusters)
    u_new = singleton.value(A, [node for cl in clusters for node in cl.nodes] + [v_new]) - lambda_t
    payoffs.append(u_new)

    probs = softmax(payoffs, beta)
    choice = np.random.choice(len(probs), p=probs)
    if choice == len(clusters):
        clusters.append(singleton)
    else:
        clusters[choice].add_node(v_new)

    nodes = [node for c in clusters for node in c.nodes]
    return refine_clusters(A, nodes, clusters, beta, lambda_base, lambda_growth, lambda_decay, max_iter=K)

def remove_node_and_resettle_soft(A, clusters, node_id, beta=3.0, lambda_base=0.1, lambda_growth=0.01, lambda_decay=0.05, K=5):
    """
    Simulates node deletion by removing all edges and excluding the node from cluster refinement.
    """
    # Remove edges of the node
    A[node_id, :] = 0
    A[:, node_id] = 0

    # Remove the node from its cluster
    node_to_remove = None
    for c in clusters:
        for node in c.nodes:
            if node.id == node_id:
                node_to_remove = node
                break
        if node_to_remove:
            break

    if node_to_remove:
        node_to_remove.cluster.remove_node(node_to_remove)

    # Remove empty clusters
    clusters = [c for c in clusters if len(c.nodes) > 0]

    # Exclude removed node from node list
    nodes = [node for c in clusters for node in c.nodes if node.id != node_id]

    # Run refinement on remaining nodes
    return refine_clusters(A, nodes, clusters, beta, lambda_base, lambda_growth, lambda_decay, max_iter=K)

def run_single_add_remove_experiment(A, y_true, clusters, beta, lambda_base, lambda_growth, lambda_decay, iter_counts=[5, 10, 20]):
    results = {
        'add': [],
        'remove': []
    }

    # 기존 노드와 클러스터 복사
    A_orig = np.copy(A)
    clusters_orig = copy.deepcopy(clusters)
    all_nodes = [node for c in clusters for node in c.nodes]
    num_original_nodes = len(all_nodes)

    # ▶ 노드 추가 실험
    print("▶ Structured Node Addition Experiment")

    target_node = np.random.choice(all_nodes)
    target_id = target_node.id
    neighbor_ids = np.nonzero(A_orig[target_id])[0]
    connection_row = np.zeros(num_original_nodes)
    connection_row[neighbor_ids] = 1

    gt_cluster = y_true[target_id]
    new_ground_truth = np.append(y_true, gt_cluster)

    for K in iter_counts:
        A_copy = np.copy(A_orig)
        clusters_copy = copy.deepcopy(clusters_orig)
        updated_clusters = add_node_and_resettle(A_copy, clusters_copy, beta, lambda_base, lambda_growth, lambda_decay, 0, connection_row, K=K)
        pred_labels = extract_cluster_labels(updated_clusters, len(new_ground_truth))
        nmi = normalized_mutual_info_score(new_ground_truth, pred_labels)
        results['add'].append((K, nmi))
        print(f"NMI after adding one node and {K} iters: {nmi:.4f}")

    # # ▶ 노드 삭제 실험
    # print("\n▶ Node Removal Experiment")

    # target_id = np.random.choice(num_original_nodes)
    # removed_ground_truth = np.delete(y_true, target_id)

    # for K in iter_counts:
    #     A_copy = np.delete(np.delete(A_orig, target_id, axis=0), target_id, axis=1)
    #     clusters_copy = copy.deepcopy(clusters_orig)
    #     updated_clusters = remove_node_and_resettle_soft(A_copy, clusters_copy, node_id=target_id, beta=beta, lambda_base=lambda_base,
    #                                                 lambda_growth=lambda_growth, lambda_decay=lambda_decay, K=K)
    #     pred_labels = extract_cluster_labels(updated_clusters, len(removed_ground_truth))
    #     nmi = normalized_mutual_info_score(removed_ground_truth, pred_labels)
    #     results['remove'].append((K, nmi))
    #     print(f"NMI after removing one node and {K} iters: {nmi:.4f}")

    return results

# # Test 1 : Zachary Karate Club (실세계 네트워크)

# import networkx as nx
# import numpy as np
# from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# # 1. Load Zachary's Karate Club graph
# G = nx.karate_club_graph()
# A = nx.to_numpy_array(G)

# # 2. Ground truth labels: based on known split
# ground_truth = [0 if G.nodes[i]['club'] == 'Mr. Hi' else 1 for i in range(len(G))]

# # 3. Run your clustering algorithm
# num_iteration = 5
# sum_nmi = 0
# for i in range(num_iteration):
#     clusters = clustering_algorithm(A, beta=3.0, lambda_base=0.1, lambda_growth=0.0005, lambda_decay=0.0005, max_iter=100)

#     # 4. Map each node to predicted cluster ID
#     label_map = {}
#     for cluster_id, cluster in enumerate(clusters):
#         for node in cluster.nodes:
#             label_map[node.id] = cluster_id
#     predicted_labels = [label_map[i] for i in range(len(A))]

#     # 5. Evaluation
#     ari = adjusted_rand_score(ground_truth, predicted_labels)
#     nmi = normalized_mutual_info_score(ground_truth, predicted_labels)

#     sum_nmi += nmi

#     print(f"\\n===== Zachary Karate Club Results =====")
#     print(f"ARI (Adjusted Rand Index): {ari:.4f}")
#     print(f"NMI (Normalized Mutual Info): {nmi:.4f}")
#     print(f"Number of Clusters Found: {len(clusters)}")

# print(sum_nmi / num_iteration)

# # Test 2 : 2-block synthetic graph (이상적인 구조)

# import numpy as np
# import networkx as nx
# from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# # 1. Synthetic 2-block graph 생성
# def generate_2block_graph(n1=15, n2=15, p_in=0.9, p_out=0.05):
#     sizes = [n1, n2]
#     probs = [[p_in, p_out], [p_out, p_in]]
#     G = nx.stochastic_block_model(sizes, probs)
#     A = nx.to_numpy_array(G)
#     ground_truth = [0]*n1 + [1]*n2
#     return A, ground_truth

# # 2. 데이터 생성
# A, ground_truth = generate_2block_graph()

# num_iteration = 5
# sum_nmi = 0

# for i in range(num_iteration):

#     # 3. 알고리즘 실행
#     clusters = clustering_algorithm(A, beta=3.0, lambda_base=0.1, lambda_growth=0.0005, lambda_decay=0.0005, max_iter=100)

#     # 4. 결과 평가
#     label_map = {}
#     for cluster_id, cluster in enumerate(clusters):
#         for node in cluster.nodes:
#             label_map[node.id] = cluster_id
#     predicted_labels = [label_map[i] for i in range(len(A))]

#     ari = adjusted_rand_score(ground_truth, predicted_labels)
#     nmi = normalized_mutual_info_score(ground_truth, predicted_labels)

#     sum_nmi += nmi

#     print(f"\\n===== 2-Block Synthetic Graph Results =====")
#     print(f"ARI: {ari:.4f}, NMI: {nmi:.4f}, Num Clusters: {len(clusters)}")

# print(sum_nmi / num_iteration)

# # Test 3 : Football Network

# import networkx as nx
# import numpy as np
# from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# # Load football network
# G = nx.read_gml("football.gml")
# A = nx.to_numpy_array(G)

# # Extract ground truth conference labels
# label_dict = {n: G.nodes[n]['value'] for n in G.nodes}
# nodes_sorted = sorted(G.nodes)
# ground_truth = [label_dict[n] for n in nodes_sorted]

# num_iteration = 5
# sum_nmi = 0

# for i in range(num_iteration):

#     # Run your clustering algorithm
#     clusters = clustering_algorithm(
#         A,
#         beta=3.0,
#         lambda_base=0.1,
#         lambda_growth=0.0005,
#         lambda_decay=0.0005,
#         max_iter=100
#     )

#     # Map predicted labels
#     label_map = {}
#     for cluster_id, cluster in enumerate(clusters):
#         for node in cluster.nodes:
#             label_map[node.id] = cluster_id
#     predicted_labels = [label_map[i] for i in range(len(A))]

#     # Evaluate
#     ari = adjusted_rand_score(ground_truth, predicted_labels)
#     nmi = normalized_mutual_info_score(ground_truth, predicted_labels)

#     sum_nmi += nmi

#     print(f"\n===== Football Network Results =====")
#     print(f"ARI: {ari:.4f}, NMI: {nmi:.4f}, Num Clusters: {len(clusters)}")

# print(sum_nmi / num_iteration)

# # Test 4 : Dolphin

# import networkx as nx
# import numpy as np
# from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# def load_dolphin_graph():
#     edges = []
#     with open("dolphin.dat") as f:
#         for line in f:
#             u, v = map(int, line.strip().split())
#             edges.append((u, v))

#     G = nx.Graph()
#     G.add_edges_from(edges)
#     A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
#     return A, G

# def load_dolphin_ground_truth(G):
#     labels = {}
#     with open("dolphin_ground_truth.dat") as f:
#         for line in f:
#             node, label = map(int, line.strip().split())
#             labels[node] = label

#     # G의 정렬된 노드 순서대로 정렬
#     ground_truth = [labels[node] for node in sorted(G.nodes())]
#     return ground_truth

# A, G = load_dolphin_graph()
# ground_truth = load_dolphin_ground_truth(G)

# num_iteration = 5
# sum_nmi = 0

# for i in range(num_iteration):

#     clusters = clustering_algorithm(
#         A,
#         beta=3.0,
#         lambda_base=0.1,
#         lambda_growth=0.0005,
#         lambda_decay=0.0005,
#         max_iter=100
#     )

#     predicted_labels = extract_cluster_labels(clusters, len(G.nodes()))

#     # 평가
#     ari = adjusted_rand_score(ground_truth, predicted_labels)
#     nmi = normalized_mutual_info_score(ground_truth, predicted_labels)

#     sum_nmi += nmi

#     print("\n===== Dolphin Network Clustering Results =====")
#     print(f"ARI: {ari:.4f}")
#     print(f"NMI: {nmi:.4f}")
#     print(f"Number of Clusters Found: {len(clusters)}")

# print(sum_nmi / num_iteration)



# parameter test 1 : lamda_base

# nmi_list = []
# for base in [0.0, 0.2, 0.4, 0.6, 0.8]:
#     nmi_value = 0
#     for i in range(5):
#         clusters = clustering_algorithm(
#             A,
#             beta=3.0,
#             lambda_base=base,
#             lambda_growth=0.0,
#             lambda_decay=0.0,
#             max_iter=100
#         )

#         label_map = {}
#         for cluster_id, cluster in enumerate(clusters):
#             for node in cluster.nodes:
#                 label_map[node.id] = cluster_id
#         # predicted_labels = [label_map[i] for i in range(len(A))]  # for graph 1, 2, 3
#         predicted_labels = extract_cluster_labels(clusters, len(G.nodes()))   # for graph 4

#         ari = adjusted_rand_score(ground_truth, predicted_labels)
#         nmi = normalized_mutual_info_score(ground_truth, predicted_labels)
#         print(f"\n===== λ_base = {base:.2f} =====")
#         print(f"ARI: {ari:.4f}, NMI: {nmi:.4f}, Num Clusters: {len(clusters)}")
#         nmi_value += nmi
    
#     nmi_list.append(nmi_value / 5)

# print("nmi_list : ", nmi_list)

# # parameter test 2 : lamda_growth and lamda_decay

# nmi_list = []
# for growth, decay in [(0.000, 0.000), (0.000, 0.0005), (0.0005, 0.000), (0.0005, 0.0005)]:
#     nmi_value = 0
#     for i in range(5):
#         clusters = clustering_algorithm(
#             A,
#             beta=3.0,
#             lambda_base=0.1,
#             lambda_growth=growth,
#             lambda_decay=decay,
#             max_iter=100
#         )

#         label_map = {}
#         for cluster_id, cluster in enumerate(clusters):
#             for node in cluster.nodes:
#                 label_map[node.id] = cluster_id
#         predicted_labels = [label_map[i] for i in range(len(A))]  # for graph 1, 2, 3
#         # predicted_labels = extract_cluster_labels(clusters, len(G.nodes()))   # for graph 4

#         ari = adjusted_rand_score(ground_truth, predicted_labels)
#         nmi = normalized_mutual_info_score(ground_truth, predicted_labels)
#         print(f"\n===== lamda_growth and lamda_decay = {growth:.3f}, {decay:.3f} =====")
#         print(f"ARI: {ari:.4f}, NMI: {nmi:.4f}, Num Clusters: {len(clusters)}")
#         nmi_value += nmi
    
#     nmi_list.append(nmi_value / 5)

# print("nmi_list : ", nmi_list)


# Dynamic graph test

# 1. Load Zachary's Karate Club graph
G = nx.karate_club_graph()
A = nx.to_numpy_array(G)

# 2. Ground truth labels: based on known split
ground_truth = [0 if G.nodes[i]['club'] == 'Mr. Hi' else 1 for i in range(len(G))]

clusters = clustering_algorithm(A, beta=3.0, lambda_base=0.1, lambda_growth=0.0005, lambda_decay=0.0005, max_iter=100)

# 4. Map each node to predicted cluster ID
label_map = {}
for cluster_id, cluster in enumerate(clusters):
    for node in cluster.nodes:
        label_map[node.id] = cluster_id
predicted_labels = [label_map[i] for i in range(len(A))]

# 5. Evaluation
nmi = normalized_mutual_info_score(ground_truth, predicted_labels)
print(nmi)

results = run_single_add_remove_experiment(
    A, ground_truth, clusters,
    beta=3.0, lambda_base=0.1, lambda_growth=0.0005, lambda_decay=0.0005, iter_counts=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
)