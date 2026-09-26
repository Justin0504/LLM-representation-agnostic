"""Novelty Benchmark: 10 concepts × 3 instances × 6 forms = 180 stimuli.

Purpose: an INDEPENDENT concept inventory to test whether the FARS geometry
found on TriForm generalizes to concepts not in the original construction set.
Directly answers Reviewer 3's benchmark-induced-geometry concern.

Design principles:
  - Concept domains DISTINCT from TriForm (arith / logic / relational / causal /
    spatial). Here we use: temporal, probabilistic, graph, data-structure,
    quantifier.
  - Template structures deliberately different: no "if X then Y", no
    "compute expression", no set operations.
  - Six surface forms matching TriForm's for direct comparison:
      en_prose, zh_prose, fr_prose, py_code, math_notation, structured_list.

Concept inventory (id : name : domain):
  0  event_ordering      : temporal
  1  duration_arithmetic : temporal
  2  conditional_probability : probabilistic
  3  expected_value      : probabilistic
  4  shortest_path       : graph
  5  connectivity        : graph
  6  stack_lifo          : data_structure
  7  tree_lookup         : data_structure
  8  universal_check     : quantifier
  9  existential_check   : quantifier
"""
from __future__ import annotations
import os, sys
from dataclasses import dataclass
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.stimuli import Stimulus

FORM_NAMES = ["en_prose", "zh_prose", "fr_prose", "py_code", "math_notation", "structured_list"]

CONCEPT_NAMES = [
    "event_ordering", "duration_arithmetic",
    "conditional_probability", "expected_value",
    "shortest_path", "connectivity",
    "stack_lifo", "tree_lookup",
    "universal_check", "existential_check",
]
DOMAIN_OF = {
    0: "temporal", 1: "temporal",
    2: "probabilistic", 3: "probabilistic",
    4: "graph", 5: "graph",
    6: "data_structure", 7: "data_structure",
    8: "quantifier", 9: "quantifier",
}

# Each concept has three instances written across six forms.
# Text below is hand-authored per (concept, instance, form) for maximum
# semantic equivalence and stylistic naturalness within each form.

STIMULI: list[tuple[int, str, dict]] = [
    # ---------- concept 0: event_ordering (temporal) ----------
    (0, "en_prose",
     {"i0": "Anna arrived at the cafe at 3:00 pm. Ben arrived thirty minutes later. Carl arrived after Ben but before 4:00 pm. In the order they arrived, the sequence is Anna, Ben, Carl.",
      "i1": "The rain started before the game began, and the game finished before dinner. Therefore the rain started before dinner.",
      "i2": "The library closes after the museum but before the coffee shop. Ranking closing times from earliest to latest gives museum, library, coffee shop."}),
    (0, "zh_prose",
     {"i0": "安娜下午3点到达咖啡馆。本三十分钟后到达。卡尔在本之后到达，但在下午4点之前。按到达顺序排列为：安娜、本、卡尔。",
      "i1": "下雨在比赛开始前发生，比赛在晚餐前结束。因此下雨在晚餐之前发生。",
      "i2": "图书馆比博物馆晚关门，但比咖啡店早关门。按关门时间从早到晚排列为：博物馆、图书馆、咖啡店。"}),
    (0, "fr_prose",
     {"i0": "Anna est arrivée au café à 15h00. Ben est arrivé trente minutes plus tard. Carl est arrivé après Ben mais avant 16h00. Dans l'ordre d'arrivée : Anna, Ben, Carl.",
      "i1": "La pluie a commencé avant le début du match, et le match s'est terminé avant le dîner. Donc la pluie a commencé avant le dîner.",
      "i2": "La bibliothèque ferme après le musée mais avant le café. Ordre des fermetures du plus tôt au plus tard : musée, bibliothèque, café."}),
    (0, "py_code",
     {"i0": "events = [('anna', 15.0), ('ben', 15.5), ('carl', 15.75)]\nordered = sorted(events, key=lambda x: x[1])\nassert [e[0] for e in ordered] == ['anna', 'ben', 'carl']",
      "i1": "rain_start < game_start < game_end < dinner_start\nassert rain_start < dinner_start  # transitivity of temporal order",
      "i2": "closing = {'museum': 17.0, 'library': 18.0, 'coffee': 19.0}\nordered = sorted(closing.items(), key=lambda x: x[1])\nassert [name for name, _ in ordered] == ['museum', 'library', 'coffee']"}),
    (0, "math_notation",
     {"i0": "Let t_A = 15:00, t_B = t_A + 30 min, t_C in (t_B, 16:00). Order by time: t_A < t_B < t_C.",
      "i1": "Given t(rain) < t(game_start) and t(game_end) < t(dinner). By transitivity of <, t(rain) < t(dinner).",
      "i2": "Let c_M, c_L, c_C denote closing times with c_M < c_L < c_C. Sorted ascending: (M, L, C)."}),
    (0, "structured_list",
     {"i0": "Task: order events by arrival time\nEvents: Anna@15:00, Ben@15:30, Carl@15:45\nResult: [Anna, Ben, Carl]",
      "i1": "Given: rain-before-game, game-before-dinner\nQuery: rain vs dinner\nAnswer: rain before dinner",
      "i2": "Sort key: closing time\nItems: museum=17:00, library=18:00, coffee=19:00\nSorted: [museum, library, coffee]"}),

    # ---------- concept 1: duration_arithmetic (temporal) ----------
    (1, "en_prose",
     {"i0": "A meeting starts at 9:45 am and lasts one hour and forty minutes. It ends at 11:25 am.",
      "i1": "A movie of two hours and fifteen minutes begins at 7:30 pm. It ends at 9:45 pm.",
      "i2": "A shift begins at 8:00 am and ends at 4:30 pm, with a thirty-minute break. The worked duration is eight hours."}),
    (1, "zh_prose",
     {"i0": "一场会议在上午9:45开始，持续1小时40分钟。会议在上午11:25结束。",
      "i1": "一部时长2小时15分钟的电影在晚上7:30开始，将在9:45结束。",
      "i2": "一个班次在上午8:00开始，下午4:30结束，包括30分钟休息，工作时长为8小时。"}),
    (1, "fr_prose",
     {"i0": "Une réunion commence à 9h45 et dure une heure quarante. Elle se termine à 11h25.",
      "i1": "Un film de deux heures quinze commence à 19h30 et se termine à 21h45.",
      "i2": "Un poste commence à 8h00 et finit à 16h30, avec une pause de trente minutes. Durée travaillée : huit heures."}),
    (1, "py_code",
     {"i0": "from datetime import time, timedelta, datetime\nstart = datetime(2024,1,1,9,45)\nend = start + timedelta(hours=1, minutes=40)\nassert end.strftime('%H:%M') == '11:25'",
      "i1": "start = datetime(2024,1,1,19,30)\nduration = timedelta(hours=2, minutes=15)\nassert (start + duration).strftime('%H:%M') == '21:45'",
      "i2": "worked = timedelta(hours=8, minutes=30) - timedelta(minutes=30)\nassert worked == timedelta(hours=8)"}),
    (1, "math_notation",
     {"i0": "t_end = t_start + Δt, where t_start = 9:45 and Δt = 1h40m. Therefore t_end = 11:25.",
      "i1": "t_end = 19:30 + 02:15 = 21:45.",
      "i2": "T_work = T_shift − T_break = (16:30 − 08:00) − 0:30 = 8:00."}),
    (1, "structured_list",
     {"i0": "Task: add duration to start time\nInputs: start=09:45, duration=1h40m\nResult: 11:25",
      "i1": "Task: end time of movie\nStart: 19:30 | Duration: 2h15m\nEnd: 21:45",
      "i2": "Task: compute worked hours\nShift: 08:00–16:30 | Break: 30m\nWorked: 8h"}),

    # ---------- concept 2: conditional_probability (probabilistic) ----------
    (2, "en_prose",
     {"i0": "A jar has three red balls and two blue balls. Without replacement, the probability that the second draw is red given the first draw was red is 2/4.",
      "i1": "In a class where 60% study math and 40% of math students also study physics, the probability that a math student studies physics is 0.4.",
      "i2": "If it rains, the probability of the picnic being cancelled is 0.9. Given that the picnic was cancelled, we cannot directly conclude it rained without knowing P(rain)."}),
    (2, "zh_prose",
     {"i0": "一个罐子里有3个红球和2个蓝球。不放回抽取时，在第一次抽到红球的条件下，第二次抽到红球的概率是2/4。",
      "i1": "一个班级中60%的学生学数学，其中40%同时学物理。给定学生学数学的条件下，学物理的概率为0.4。",
      "i2": "如果下雨，野餐被取消的概率为0.9。仅知道野餐被取消，无法直接推断下雨，除非知道P(下雨)。"}),
    (2, "fr_prose",
     {"i0": "Un pot contient trois boules rouges et deux boules bleues. Sans remise, la probabilité que le deuxième tirage soit rouge sachant que le premier était rouge est 2/4.",
      "i1": "Dans une classe où 60 % étudient les maths et 40 % d'entre eux étudient aussi la physique, la probabilité qu'un étudiant en maths étudie la physique est 0,4.",
      "i2": "S'il pleut, la probabilité que le pique-nique soit annulé est 0,9. Sachant qu'il a été annulé, on ne peut pas conclure directement qu'il a plu sans connaître P(pluie)."}),
    (2, "py_code",
     {"i0": "P_A = 3/5  # first red\nP_B_given_A = 2/4  # second red given first red\nassert abs(P_B_given_A - 0.5) < 1e-9",
      "i1": "P_math = 0.60\nP_phys_given_math = 0.40\nassert P_phys_given_math == 0.4",
      "i2": "P_cancel_given_rain = 0.9\n# P(rain | cancel) requires P(rain) via Bayes\nassert 'P(rain)' in 'need P(rain) for inverse conditional'"}),
    (2, "math_notation",
     {"i0": "P(B_2 = red | B_1 = red) = 2/4 (draws without replacement, jar: 3R, 2B).",
      "i1": "P(physics | math) = 0.4, given P(math) = 0.6.",
      "i2": "P(cancel | rain) = 0.9. P(rain | cancel) = P(cancel|rain) P(rain) / P(cancel), requires P(rain)."}),
    (2, "structured_list",
     {"i0": "Setup: jar={red:3, blue:2}, no replacement\nEvent A: first=red | Event B: second=red\nP(B|A) = 2/4",
      "i1": "Population: 60% math | 40% of math also physics\nGiven: student in math\nP(physics|math) = 0.40",
      "i2": "Prior: P(cancel|rain) = 0.9\nQuery: P(rain|cancel)\nAnswer: needs P(rain); apply Bayes"}),

    # ---------- concept 3: expected_value (probabilistic) ----------
    (3, "en_prose",
     {"i0": "A fair six-sided die is rolled. The expected value of the roll is 3.5.",
      "i1": "A lottery pays $10 with probability 0.1 and $0 otherwise. The expected payout is $1.",
      "i2": "A coin flip pays $3 on heads and −$1 on tails. The expected payoff is $1."}),
    (3, "zh_prose",
     {"i0": "投掷一个均匀的六面骰子，点数的期望值是3.5。",
      "i1": "彩票以0.1的概率支付10美元，否则支付0美元，期望回报为1美元。",
      "i2": "抛硬币正面得3美元，反面损失1美元，期望收益为1美元。"}),
    (3, "fr_prose",
     {"i0": "Un dé équilibré à six faces est lancé. L'espérance mathématique du résultat est 3,5.",
      "i1": "Une loterie paie 10 $ avec probabilité 0,1 et 0 $ sinon. Le gain espéré est de 1 $.",
      "i2": "Un lancer de pièce paie 3 $ sur pile et −1 $ sur face. L'espérance du gain est 1 $."}),
    (3, "py_code",
     {"i0": "E = sum(i * (1/6) for i in range(1, 7))\nassert abs(E - 3.5) < 1e-9",
      "i1": "E = 10 * 0.1 + 0 * 0.9\nassert E == 1.0",
      "i2": "E = 3 * 0.5 + (-1) * 0.5\nassert E == 1.0"}),
    (3, "math_notation",
     {"i0": "E[X] = Σ_{i=1..6} i · (1/6) = (1+2+3+4+5+6)/6 = 3.5.",
      "i1": "E[X] = 10 · 0.1 + 0 · 0.9 = 1.",
      "i2": "E[X] = 3 · 0.5 + (−1) · 0.5 = 1."}),
    (3, "structured_list",
     {"i0": "Task: expected roll of fair d6\nOutcomes: {1..6}, each p=1/6\nE = 3.5",
      "i1": "Task: expected lottery payout\nPayouts: {$10 w.p. 0.1, $0 w.p. 0.9}\nE = $1",
      "i2": "Task: expected coin payoff\nH: +$3 (0.5) | T: -$1 (0.5)\nE = $1"}),

    # ---------- concept 4: shortest_path (graph) ----------
    (4, "en_prose",
     {"i0": "Three cities A, B, and C. A-B has distance 4 and B-C has distance 3, while A-C direct has distance 10. The shortest path from A to C is A-B-C with total distance 7.",
      "i1": "In a graph with nodes 1, 2, 3, 4 and unit edges 1-2, 2-3, 3-4, the shortest path from 1 to 4 has length 3.",
      "i2": "Two routes from X to Y: X-P-Y (length 5+6) and X-Q-Y (length 4+8). The shortest is X-P-Y at length 11."}),
    (4, "zh_prose",
     {"i0": "三个城市A、B、C。A-B距离为4，B-C距离为3，A-C直接距离为10。A到C的最短路径是A-B-C，总距离为7。",
      "i1": "在节点为1,2,3,4，单位边为1-2、2-3、3-4的图中，从1到4的最短路径长度为3。",
      "i2": "从X到Y的两条路径：X-P-Y (长5+6) 和 X-Q-Y (长4+8)。最短是X-P-Y，长度为11。"}),
    (4, "fr_prose",
     {"i0": "Trois villes A, B, C. A-B a la distance 4 et B-C a la distance 3, tandis que A-C direct a la distance 10. Le plus court chemin de A à C est A-B-C avec la distance totale 7.",
      "i1": "Dans un graphe avec les nœuds 1, 2, 3, 4 et les arêtes unitaires 1-2, 2-3, 3-4, le plus court chemin de 1 à 4 est de longueur 3.",
      "i2": "Deux itinéraires de X à Y : X-P-Y (longueur 5+6) et X-Q-Y (longueur 4+8). Le plus court est X-P-Y, longueur 11."}),
    (4, "py_code",
     {"i0": "graph = {'A': {'B': 4, 'C': 10}, 'B': {'A': 4, 'C': 3}, 'C': {'A': 10, 'B': 3}}\ncost_ABC = graph['A']['B'] + graph['B']['C']\nassert cost_ABC == 7 and cost_ABC < graph['A']['C']",
      "i1": "edges = [(1,2), (2,3), (3,4)]\n# unit weights, chain\nassert len(edges) == 3  # shortest path 1->4 length 3",
      "i2": "routes = {'XPY': 5+6, 'XQY': 4+8}\nshortest = min(routes.items(), key=lambda x: x[1])\nassert shortest == ('XPY', 11)"}),
    (4, "math_notation",
     {"i0": "d(A, C) = min(d(A,C), d(A,B) + d(B,C)) = min(10, 4+3) = 7.",
      "i1": "In the path graph P_4 with unit weights, d(1, 4) = 3.",
      "i2": "d(X, Y) = min(d(X,P)+d(P,Y), d(X,Q)+d(Q,Y)) = min(11, 12) = 11."}),
    (4, "structured_list",
     {"i0": "Nodes: A, B, C | Edges: A-B:4, B-C:3, A-C:10\nShortest A→C: A-B-C, cost 7",
      "i1": "Graph: chain 1-2-3-4 (unit weights)\nShortest 1→4: length 3",
      "i2": "Routes: X-P-Y=11, X-Q-Y=12\nShortest: X-P-Y=11"}),

    # ---------- concept 5: connectivity (graph) ----------
    (5, "en_prose",
     {"i0": "In a graph on nodes A, B, C, D with edges A-B and C-D but no edge between the two pairs, nodes A and C are not connected.",
      "i1": "A graph with edges 1-2, 2-3, 3-4 forms a single connected component containing all four nodes.",
      "i2": "Removing the bridge edge M-N from a graph splits it into two components. Whether P and Q remain connected depends on which component each lies in."}),
    (5, "zh_prose",
     {"i0": "一个图有节点A、B、C、D，边为A-B和C-D，两对之间没有边。节点A和C不连通。",
      "i1": "边为1-2, 2-3, 3-4的图形成包含所有四个节点的单一连通分量。",
      "i2": "从图中删除桥梁边M-N，会将其分成两个连通分量。P和Q是否连通取决于它们分别位于哪个分量中。"}),
    (5, "fr_prose",
     {"i0": "Dans un graphe avec les nœuds A, B, C, D et les arêtes A-B et C-D, sans arête entre les deux paires, A et C ne sont pas connectés.",
      "i1": "Un graphe avec les arêtes 1-2, 2-3, 3-4 forme une seule composante connexe contenant les quatre nœuds.",
      "i2": "Supprimer l'arête pont M-N d'un graphe le sépare en deux composantes. Que P et Q restent connectés dépend de la composante de chacun."}),
    (5, "py_code",
     {"i0": "edges = {('A','B'), ('C','D')}\n# BFS from A only reaches {A, B}\nassert 'C' not in {'A','B'}",
      "i1": "edges = [(1,2), (2,3), (3,4)]\n# all nodes reachable from 1\nassert set([1,2,3,4]) == {1,2,3,4}",
      "i2": "# Removing bridge M-N splits graph\n# connected(P, Q) depends on their components\ncomponents = [{'M','P'}, {'N','Q'}]\nassert not (any({p,q} <= c for p,q in [('P','Q')] for c in components))"}),
    (5, "math_notation",
     {"i0": "E = {(A,B), (C,D)}. A ~ B, C ~ D, A !~ C (no path A→C in E).",
      "i1": "G = (V, E) with V = {1,2,3,4}, E = {(1,2),(2,3),(3,4)}. |CC(G)| = 1.",
      "i2": "Bridge e ∈ E ⇒ G \\ {e} has |CC| = |CC(G)| + 1. P ~ Q in G \\ {e} iff same CC."}),
    (5, "structured_list",
     {"i0": "Edges: {A-B, C-D}\nQuery: A connected to C?\nAnswer: No",
      "i1": "Edges: [1-2, 2-3, 3-4]\nComponents: 1\nAll nodes connected: Yes",
      "i2": "Remove bridge: M-N\nResult: 2 components\nP connected to Q: depends on component membership"}),

    # ---------- concept 6: stack_lifo (data structure) ----------
    (6, "en_prose",
     {"i0": "Push 1, push 2, push 3, then pop. The popped value is 3 because the stack is last-in-first-out.",
      "i1": "Starting from an empty stack, push A, push B, pop, push C, pop, pop. The final stack is empty.",
      "i2": "A stack after pushing 5, 6, 7 has 7 on top. Peek returns 7 without removing it."}),
    (6, "zh_prose",
     {"i0": "依次入栈1、2、3，然后弹出。弹出的值是3，因为栈是后进先出的。",
      "i1": "从空栈开始，压入A、B，弹出，压入C，弹出，弹出。最终栈为空。",
      "i2": "压入5、6、7后，栈顶是7。Peek返回7但不删除它。"}),
    (6, "fr_prose",
     {"i0": "Empiler 1, empiler 2, empiler 3, puis dépiler. La valeur dépilée est 3 car la pile est LIFO.",
      "i1": "À partir d'une pile vide : empiler A, empiler B, dépiler, empiler C, dépiler, dépiler. La pile finale est vide.",
      "i2": "Une pile après avoir empilé 5, 6, 7 a 7 au sommet. Peek renvoie 7 sans le retirer."}),
    (6, "py_code",
     {"i0": "s = []\ns.append(1); s.append(2); s.append(3)\nassert s.pop() == 3",
      "i1": "s = []\ns.append('A'); s.append('B'); s.pop()\ns.append('C'); s.pop(); s.pop()\nassert s == []",
      "i2": "s = [5, 6, 7]\nassert s[-1] == 7 and s == [5, 6, 7]  # peek does not pop"}),
    (6, "math_notation",
     {"i0": "S = (), push(1), push(2), push(3) ⇒ S = (1,2,3). pop ⇒ (return 3, S = (1,2)).",
      "i1": "S = () → push A → push B → pop → push C → pop → pop → S = ().",
      "i2": "S = (5,6,7). peek(S) = 7. |S| unchanged."}),
    (6, "structured_list",
     {"i0": "Ops: push(1), push(2), push(3), pop\nStack after: [1, 2]\nPopped: 3",
      "i1": "Ops: pushA, pushB, pop, pushC, pop, pop\nStack after: []",
      "i2": "Stack: [5, 6, 7]\nOp: peek\nReturn: 7 | Stack unchanged"}),

    # ---------- concept 7: tree_lookup (data structure) ----------
    (7, "en_prose",
     {"i0": "In a binary search tree, to find 15 we compare with the root: if root is 10, go right; if right child is 20, go left; if that child is 15, return it.",
      "i1": "A dictionary maps 'apple' to 3 and 'banana' to 5. Looking up 'banana' returns 5.",
      "i2": "A trie stores 'cat' and 'car'. Looking up 'card' traverses 'c'-'a'-'r' and fails to find a terminal 'd'."}),
    (7, "zh_prose",
     {"i0": "在二叉搜索树中查找15：与根比较，若根为10则向右；若右孩子为20则向左；若该孩子为15则返回。",
      "i1": "一个字典将'apple'映射到3，'banana'映射到5。查找'banana'返回5。",
      "i2": "一个trie存储'cat'和'car'。查找'card'走'c'-'a'-'r'路径后未找到终止的'd'。"}),
    (7, "fr_prose",
     {"i0": "Dans un ABR, pour trouver 15 : comparer à la racine ; si racine = 10, aller à droite ; si enfant droit = 20, aller à gauche ; si cet enfant = 15, le retourner.",
      "i1": "Un dictionnaire associe 'apple' à 3 et 'banana' à 5. La recherche de 'banana' renvoie 5.",
      "i2": "Un trie stocke 'cat' et 'car'. Rechercher 'card' parcourt 'c'-'a'-'r' et échoue à trouver un 'd' terminal."}),
    (7, "py_code",
     {"i0": "class Node:\n    def __init__(self, v): self.v, self.l, self.r = v, None, None\nroot = Node(10); root.r = Node(20); root.r.l = Node(15)\nassert root.r.l.v == 15",
      "i1": "d = {'apple': 3, 'banana': 5}\nassert d['banana'] == 5",
      "i2": "trie = {'c': {'a': {'t': {}, 'r': {}}}}\nnode = trie\nfor ch in 'card':\n    node = node.get(ch) if node else None\nassert node is None  # 'd' not found"}),
    (7, "math_notation",
     {"i0": "BST invariant: L(v) < v < R(v). Find(15): root=10, right, node=20, left, node=15 ✓.",
      "i1": "M = {apple ↦ 3, banana ↦ 5}. M(banana) = 5.",
      "i2": "Trie T = {cat, car}. Traverse(card) = c → a → r, T(...,d) = ⊥."}),
    (7, "structured_list",
     {"i0": "BST: root=10 → right=20 → left=15\nFind: 15\nPath: root→right→left → hit",
      "i1": "Map: {apple:3, banana:5}\nQuery: banana\nReturn: 5",
      "i2": "Trie: {cat, car}\nQuery: card\nResult: not_found at 'd'"}),

    # ---------- concept 8: universal_check (quantifier) ----------
    (8, "en_prose",
     {"i0": "All apples in the basket are red. Therefore any specific apple you pick from the basket is red.",
      "i1": "Every student in the class passed the exam. So John, who is in the class, passed the exam.",
      "i2": "All numbers in the list [4, 8, 12, 16] are even. Picking any element yields an even number."}),
    (8, "zh_prose",
     {"i0": "篮子里所有苹果都是红色的。因此从篮子里任取一个苹果都是红色的。",
      "i1": "班里每个学生都通过了考试。所以在班里的约翰通过了考试。",
      "i2": "列表[4, 8, 12, 16]中所有数都是偶数。取任意元素都是偶数。"}),
    (8, "fr_prose",
     {"i0": "Toutes les pommes du panier sont rouges. Donc toute pomme prise du panier est rouge.",
      "i1": "Chaque élève de la classe a réussi l'examen. Donc John, qui est dans la classe, a réussi.",
      "i2": "Tous les nombres de la liste [4, 8, 12, 16] sont pairs. Prendre n'importe quel élément donne un pair."}),
    (8, "py_code",
     {"i0": "basket = ['red_apple_1', 'red_apple_2', 'red_apple_3']\nassert all(a.startswith('red') for a in basket)",
      "i1": "passed = {'alice': True, 'john': True, 'bob': True}\nassert all(passed.values())",
      "i2": "nums = [4, 8, 12, 16]\nassert all(n % 2 == 0 for n in nums)"}),
    (8, "math_notation",
     {"i0": "∀ a ∈ Basket, red(a). Given b ∈ Basket, red(b).",
      "i1": "∀ s ∈ Class, passed(s). John ∈ Class ⇒ passed(John).",
      "i2": "∀ n ∈ {4, 8, 12, 16}, n mod 2 = 0. Any element is even."}),
    (8, "structured_list",
     {"i0": "Set: basket of apples\nProperty: all red\nQuery: pick 1, is red? → Yes",
      "i1": "Set: class students\nProperty: all passed\nQuery: John passed? → Yes",
      "i2": "Set: [4, 8, 12, 16]\nProperty: all even\nQuery: any element even? → Yes"}),

    # ---------- concept 9: existential_check (quantifier) ----------
    (9, "en_prose",
     {"i0": "At least one book on the shelf is about physics. Therefore some book on the shelf is not a novel.",
      "i1": "There exists a prime number between 20 and 30. Namely, 23 is prime.",
      "i2": "Some element of [1, 4, 7, 9] is even. Namely, 4 satisfies this."}),
    (9, "zh_prose",
     {"i0": "书架上至少有一本关于物理的书。因此书架上有某本书不是小说。",
      "i1": "存在一个介于20和30之间的素数。例如，23是素数。",
      "i2": "[1, 4, 7, 9]中存在偶数。例如，4满足条件。"}),
    (9, "fr_prose",
     {"i0": "Au moins un livre sur l'étagère porte sur la physique. Donc certains livres ne sont pas des romans.",
      "i1": "Il existe un nombre premier entre 20 et 30. En effet, 23 est premier.",
      "i2": "Certains éléments de [1, 4, 7, 9] sont pairs. En effet, 4 satisfait."}),
    (9, "py_code",
     {"i0": "shelf = ['novel', 'physics_text', 'novel']\nassert any('physics' in b for b in shelf)",
      "i1": "def is_prime(n):\n    return n > 1 and all(n % k for k in range(2, int(n**0.5)+1))\nassert any(is_prime(n) for n in range(20, 31))",
      "i2": "nums = [1, 4, 7, 9]\nassert any(n % 2 == 0 for n in nums)"}),
    (9, "math_notation",
     {"i0": "∃ b ∈ Shelf : subject(b) = physics.",
      "i1": "∃ p ∈ (20, 30) : prime(p). Witness: 23.",
      "i2": "∃ n ∈ {1, 4, 7, 9} : n mod 2 = 0. Witness: n = 4."}),
    (9, "structured_list",
     {"i0": "Set: books on shelf\nQuery: any about physics?\nResult: Yes (witness: physics_text)",
      "i1": "Range: 20..30\nProperty: prime\nWitness: 23",
      "i2": "Set: [1, 4, 7, 9]\nProperty: even\nWitness: 4"}),
]


def generate_novelty_stimuli() -> List[Stimulus]:
    """Return all 180 Stimulus objects (10 concepts × 3 instances × 6 forms)."""
    out: List[Stimulus] = []
    for concept_id, form, texts in STIMULI:
        form_id = FORM_NAMES.index(form)
        for inst_key, text in sorted(texts.items()):  # i0, i1, i2
            out.append(Stimulus(
                concept_id=concept_id,
                form_id=form_id,
                concept_name=CONCEPT_NAMES[concept_id],
                form_name=form,
                text=text,
            ))
    return out


if __name__ == "__main__":
    stim = generate_novelty_stimuli()
    print(f"Total stimuli: {len(stim)}")
    print(f"Concepts: {sorted(set(s.concept_id for s in stim))}")
    print(f"Forms: {sorted(set(s.form_id for s in stim))}")
    print(f"Example: {stim[0]}")
