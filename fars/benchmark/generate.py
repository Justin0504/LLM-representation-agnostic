"""
Programmatic stimulus generation for the TriForm Benchmark.

Generates semantically equivalent reasoning problems across 6 surface forms:
  - en_prose: English natural language
  - zh_prose: Chinese natural language
  - fr_prose: French natural language
  - py_code: Python code
  - math_notation: Mathematical/formal notation
  - structured_list: Structured data (JSON-like / step-by-step)

Each concept × instance × form triple produces one Stimulus.

Design: 20 concepts × 3 instances × 6 forms = 360 stimuli
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.stimuli import Stimulus

# ---------------------------------------------------------------------------
# Form generators: each takes a concept_id, instance dict, and concept metadata
# and returns the text in the target form.
# ---------------------------------------------------------------------------

FORM_NAMES = ["en_prose", "zh_prose", "fr_prose", "py_code", "math_notation", "structured_list"]
FORM_CATEGORIES = {
    "en_prose": "natural", "zh_prose": "natural", "fr_prose": "natural",
    "py_code": "formal", "math_notation": "formal", "structured_list": "formal",
}

# --- Arithmetic generators ---

def _arith_multi_step(inst, form):
    expr = inst["canonical"].replace("Compute ", "")
    ans = inst["answer"]
    if form == "en_prose":
        return f"Calculate the value of the expression {expr}. First evaluate any operations inside parentheses, then multiply and divide from left to right, and finally add and subtract. The answer is {ans}."
    elif form == "zh_prose":
        return f"计算表达式 {expr} 的值。首先计算括号内的运算，然后从左到右执行乘除法，最后执行加减法。答案是 {ans}。"
    elif form == "fr_prose":
        return f"Calculez la valeur de l'expression {expr}. Évaluez d'abord les opérations entre parenthèses, puis multipliez et divisez de gauche à droite, et enfin additionnez et soustrayez. La réponse est {ans}."
    elif form == "py_code":
        return f"# Evaluate arithmetic expression\nresult = {expr}\nassert result == {ans}  # multi-step evaluation"
    elif form == "math_notation":
        return f"Let E = {expr}. By order of operations (PEMDAS): E = {ans}."
    elif form == "structured_list":
        return f"Task: Evaluate expression\nInput: {expr}\nStep 1: Evaluate parentheses\nStep 2: Multiply and divide left to right\nStep 3: Add and subtract left to right\nResult: {ans}"

def _arith_modular(inst, form):
    parts = inst["canonical"].replace("?", "").split()
    # "What is 47 mod 7?"
    a, b = int(parts[2]), int(parts[4])
    ans = inst["answer"]
    if form == "en_prose":
        return f"When {a} is divided by {b}, the quotient is {a // b} and the remainder is {ans}. Therefore {a} modulo {b} equals {ans}."
    elif form == "zh_prose":
        return f"将 {a} 除以 {b}，商为 {a // b}，余数为 {ans}。因此 {a} 模 {b} 等于 {ans}。"
    elif form == "fr_prose":
        return f"Lorsque {a} est divisé par {b}, le quotient est {a // b} et le reste est {ans}. Donc {a} modulo {b} est égal à {ans}."
    elif form == "py_code":
        return f"# Modular arithmetic\nresult = {a} % {b}\nassert result == {ans}  # remainder of {a} / {b}"
    elif form == "math_notation":
        return f"{a} ≡ {ans} (mod {b}), since {a} = {b} × {a // b} + {ans}."
    elif form == "structured_list":
        return f"Task: Modular arithmetic\nDividend: {a}\nDivisor: {b}\nQuotient: {a // b}\nRemainder: {ans}\nResult: {a} mod {b} = {ans}"

def _arith_proportional(inst, form):
    can = inst["canonical"]
    ans = inst["answer"]
    if form == "en_prose":
        return f"{can} Using proportional reasoning, if the ratio is maintained, the answer is {ans}."
    elif form == "zh_prose":
        if "items cost" in can:
            return f"如果3件物品花费12元，那么7件物品花费多少？按比例计算：单价为4元，7件物品花费 {ans} 元。"
        elif "workers" in can:
            return f"如果5个工人10天完成工作，2个工人需要多少天？工作总量不变，{ans} 天。"
        else:
            return f"如果8升液体能装满2个罐子，5个罐子需要多少升？按比例计算，答案是 {ans} 升。"
    elif form == "fr_prose":
        return f"{can} Par raisonnement proportionnel, si le rapport est maintenu, la réponse est {ans}."
    elif form == "py_code":
        if "items cost" in can:
            return f"# Proportional reasoning\nunit_price = 12 / 3\nresult = unit_price * 7\nassert result == {ans}"
        elif "workers" in can:
            return f"# Inverse proportion: workers × days = constant\ntotal_work = 5 * 10\nresult = total_work / 2\nassert result == {ans}"
        else:
            return f"# Direct proportion\nunit = 8 / 2\nresult = unit * 5\nassert result == {ans}"
    elif form == "math_notation":
        if "items cost" in can:
            return f"Let c = cost per item = 12/3 = 4. For 7 items: 7c = 7 × 4 = {ans}."
        elif "workers" in can:
            return f"Let W = total work = 5 × 10 = 50 worker-days. For 2 workers: t = W/2 = {ans} days."
        else:
            return f"Let v = volume per tank = 8/2 = 4 L. For 5 tanks: 5v = {ans} L."
    elif form == "structured_list":
        return f"Task: Proportional reasoning\nGiven: {can}\nMethod: Maintain constant ratio\nAnswer: {ans}"

def _arith_gcd(inst, form):
    parts = inst["canonical"].replace("GCD of ", "").split(" and ")
    a, b = int(parts[0]), int(parts[1])
    ans = inst["answer"]
    if form == "en_prose":
        return f"To find the greatest common divisor of {a} and {b}, apply the Euclidean algorithm: repeatedly divide the larger number by the smaller and take the remainder, until the remainder is zero. The last non-zero remainder is the GCD, which is {ans}."
    elif form == "zh_prose":
        return f"求 {a} 和 {b} 的最大公约数，使用辗转相除法：用较大数除以较小数取余数，重复直到余数为零。最后一个非零余数即为最大公约数 {ans}。"
    elif form == "fr_prose":
        return f"Pour trouver le PGCD de {a} et {b}, appliquez l'algorithme d'Euclide : divisez successivement le plus grand par le plus petit en prenant le reste, jusqu'à obtenir un reste nul. Le dernier reste non nul est le PGCD, soit {ans}."
    elif form == "py_code":
        return f"# Euclidean algorithm for GCD\ndef gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\nassert gcd({a}, {b}) == {ans}"
    elif form == "math_notation":
        return f"gcd({a}, {b}): {a} = {b} × {a // b} + {a % b}; continue until remainder = 0. gcd({a}, {b}) = {ans}."
    elif form == "structured_list":
        return f"Task: Greatest common divisor\nInput: a={a}, b={b}\nAlgorithm: Euclidean (repeated division)\nResult: gcd({a},{b}) = {ans}"


# --- Logic generators ---

def _logic_syllogism(inst, form):
    can = inst["canonical"]
    # Parse: "All X are Y. All Z are X. Therefore all Z are Y."
    sentences = [s.strip().rstrip('.') for s in can.split('.') if s.strip()]
    p1, p2, conclusion = sentences[0], sentences[1], sentences[2]
    if form == "en_prose":
        return f"{p1}. {p2}. By the rules of categorical syllogism, we can conclude: {conclusion}. This follows from the transitivity of the 'all...are...' relation."
    elif form == "zh_prose":
        if "mammals" in can:
            return '所有哺乳动物都是温血动物。所有狗都是哺乳动物。根据三段论推理规则，所有狗都是温血动物。这是由"所有...是..."关系的传递性得出的。'
        elif "roses" in can:
            return "所有玫瑰都是花。所有花都需要水。根据三段论推理规则，所有玫瑰都需要水。"
        else:
            return "有些鸟会飞。所有鹰都是鸟。根据三段论推理规则，有些鹰会飞。"
    elif form == "fr_prose":
        if "mammals" in can:
            return "Tous les mammifères sont à sang chaud. Tous les chiens sont des mammifères. Par syllogisme catégorique, tous les chiens sont à sang chaud."
        elif "roses" in can:
            return "Toutes les roses sont des fleurs. Toutes les fleurs ont besoin d'eau. Donc, toutes les roses ont besoin d'eau."
        else:
            return "Certains oiseaux peuvent voler. Tous les aigles sont des oiseaux. Donc, certains aigles peuvent voler."
    elif form == "py_code":
        return f"# Categorical syllogism\n# Premise 1: {p1}\n# Premise 2: {p2}\ndef check_syllogism(category_A, category_B, category_C):\n    # If all A are B, and all B are C, then all A are C\n    return category_A.issubset(category_B) and category_B.issubset(category_C)"
    elif form == "math_notation":
        return f"Let A ⊆ B (Premise 1: {p1}). Let C ⊆ A (Premise 2: {p2}). By transitivity of ⊆: C ⊆ B. ∴ {conclusion}."
    elif form == "structured_list":
        return f"Task: Categorical syllogism\nPremise 1: {p1}\nPremise 2: {p2}\nRule: If A ⊆ B and B ⊆ C, then A ⊆ C\nConclusion: {conclusion}\nValid: True"

def _logic_modus_ponens(inst, form):
    can = inst["canonical"]
    sentences = [s.strip().rstrip('.') for s in can.split('.') if s.strip()]
    if form == "en_prose":
        return f"{can} This is an application of modus ponens: given P → Q and P, we conclude Q."
    elif form == "zh_prose":
        if "rains" in can:
            return "如果下雨，地面会湿。现在下雨了。根据肯定前件律（假言推理），地面一定是湿的。"
        elif "x > 5" in can:
            return "如果 x > 5，则 x > 3。已知 x = 7（满足 x > 5）。根据肯定前件律，x > 3。"
        else:
            return "如果开关打开，灯就亮。开关已打开。根据肯定前件律，灯一定亮着。"
    elif form == "fr_prose":
        if "rains" in can:
            return "S'il pleut, le sol est mouillé. Il pleut. Par modus ponens, le sol est mouillé."
        elif "x > 5" in can:
            return "Si x > 5, alors x > 3. x = 7. Par modus ponens, x > 3."
        else:
            return "Si l'interrupteur est allumé, la lumière est allumée. L'interrupteur est allumé. Donc, la lumière est allumée."
    elif form == "py_code":
        return f"# Modus ponens: P -> Q, P |- Q\ndef modus_ponens(p_implies_q, p):\n    if p and p_implies_q:\n        return True  # Q must be true\n    return None\nassert modus_ponens(True, True) == True"
    elif form == "math_notation":
        return "Given: P → Q (conditional). Given: P (antecedent holds). By modus ponens: ∴ Q."
    elif form == "structured_list":
        return f"Task: Modus ponens\nRule: P → Q\nFact: P is true\nInference: Modus ponens (affirming the antecedent)\nConclusion: Q is true\nValid: True"

def _logic_contrapositive(inst, form):
    can = inst["canonical"]
    if form == "en_prose":
        return f"{can} This uses the contrapositive: P → Q is logically equivalent to ¬Q → ¬P."
    elif form == "zh_prose":
        if "rains" in can:
            return "如果下雨则地面湿。地面没有湿。由逆否命题（P→Q 等价于 ¬Q→¬P），因此没有下雨。"
        elif "prime" in can:
            return "如果 x 是大于2的质数，则 x 是奇数。x 是偶数。由逆否命题，x 不是质数或 x ≤ 2。"
        else:
            return "如果警报响则有危险。没有危险。由逆否命题，警报没有响。"
    elif form == "fr_prose":
        if "rains" in can:
            return "Si il pleut alors le sol est mouillé. Le sol n'est pas mouillé. Par contraposée, il ne pleut pas."
        elif "prime" in can:
            return "Si x est premier et x>2, alors x est impair. x est pair. Par contraposée, x n'est pas premier ou x≤2."
        else:
            return "Si l'alarme sonne alors il y a un danger. Il n'y a pas de danger. Par contraposée, l'alarme ne sonne pas."
    elif form == "py_code":
        return "# Contrapositive: P -> Q equivalent to not Q -> not P\ndef contrapositive(p_implies_q, not_q):\n    if p_implies_q and not_q:\n        return True  # not P must hold\n    return None\nassert contrapositive(True, True) == True"
    elif form == "math_notation":
        return "Given: P → Q. Contrapositive: ¬Q → ¬P. Given: ¬Q. ∴ ¬P."
    elif form == "structured_list":
        return "Task: Contrapositive reasoning\nRule: P → Q\nEquivalent: ¬Q → ¬P\nFact: ¬Q (consequent is false)\nConclusion: ¬P (antecedent must be false)\nValid: True"

def _logic_negation(inst, form):
    can = inst["canonical"]
    if form == "en_prose":
        return f"{can} This is De Morgan's law, which governs how negation distributes over conjunction and disjunction."
    elif form == "zh_prose":
        if "A and B" in can and "not A" in can:
            return "非(A且B) 等价于 (非A)或(非B)。这是德摩根定律：否定分配到合取上时，合取变为析取。"
        elif "P or Q" in can:
            return "非(P或Q) 等价于 (非P)且(非Q)。这是德摩根定律：否定分配到析取上时，析取变为合取。"
        else:
            return "非(A且(非B)) 等价于 (非A)或B。先对合取应用德摩根定律，再对双重否定化简。"
    elif form == "fr_prose":
        if "A and B" in can and "not A" in can:
            return "Non(A et B) est équivalent à (non A) ou (non B). C'est la loi de De Morgan."
        elif "P or Q" in can:
            return "Non(P ou Q) est équivalent à (non P) et (non Q). C'est la loi de De Morgan."
        else:
            return "Non(A et (non B)) est équivalent à (non A) ou B. Par la loi de De Morgan et double négation."
    elif form == "py_code":
        return "# De Morgan's Laws\n# not (A and B) == (not A) or (not B)\n# not (A or B) == (not A) and (not B)\nassert not (True and False) == (not True) or (not False)"
    elif form == "math_notation":
        return "De Morgan's Laws: ¬(A ∧ B) ≡ ¬A ∨ ¬B; ¬(A ∨ B) ≡ ¬A ∧ ¬B."
    elif form == "structured_list":
        return "Task: De Morgan's Law\nRule 1: ¬(A ∧ B) = ¬A ∨ ¬B\nRule 2: ¬(A ∨ B) = ¬A ∧ ¬B\nApplication: Negation distributes over logical connectives, swapping ∧ and ∨"


# --- Relational generators ---

def _rel_transitivity(inst, form):
    can = inst["canonical"]
    ans = inst["answer"]
    if form == "en_prose":
        return f"{can} By transitivity of the ordering relation, {ans} is the answer."
    elif form == "zh_prose":
        if "Alice" in can:
            return f"Alice 比 Bob 高。Bob 比 Carol 高。由传递性，Alice 最高。"
        elif "X weighs" in can:
            return f"X 比 Y 重。Y 比 Z 重。由传递性，X 最重。"
        else:
            return f"A队得分高于B队。B队得分高于C队。由传递性，A队得分最高。"
    elif form == "fr_prose":
        if "Alice" in can:
            return f"Alice est plus grande que Bob. Bob est plus grand que Carol. Par transitivité, Alice est la plus grande."
        elif "X weighs" in can:
            return f"X pèse plus que Y. Y pèse plus que Z. Par transitivité, X est le plus lourd."
        else:
            return f"L'équipe A a marqué plus que B. B a marqué plus que C. Par transitivité, l'équipe A a le plus marqué."
    elif form == "py_code":
        return f"# Transitive ordering\ndef find_max(comparisons):\n    # comparisons: list of (greater, lesser) tuples\n    # By transitivity: if a > b and b > c then a > c\n    all_items = set(x for pair in comparisons for x in pair)\n    for item in all_items:\n        if all(item != lesser for _, lesser in comparisons if _ == item) or \\\n           not any(item == lesser for _, lesser in comparisons):\n            if any(item == greater for greater, _ in comparisons):\n                return item  # '{ans}'"
    elif form == "math_notation":
        return f"Given: a > b ∧ b > c. By transitivity of >: a > c. ∴ max(a,b,c) = a = {ans}."
    elif form == "structured_list":
        return f"Task: Transitive ordering\nRelation 1: A > B\nRelation 2: B > C\nRule: Transitivity (A > B ∧ B > C → A > C)\nConclusion: Maximum = {ans}"

def _rel_set_intersection(inst, form):
    can = inst["canonical"]
    ans = inst["answer"]
    if form == "en_prose":
        return f"{can} The intersection contains only those elements present in both sets. The result is {ans}."
    elif form == "zh_prose":
        return f"求两个集合的交集：找出同时属于两个集合的所有元素。结果为 {ans}。"
    elif form == "fr_prose":
        return f"L'intersection contient uniquement les éléments présents dans les deux ensembles. Le résultat est {ans}."
    elif form == "py_code":
        parts = can.split(", B = ")
        a_str = parts[0].replace("A = ", "")
        b_str = parts[1].split(".")[0]
        return f"A = {a_str}\nB = {b_str}\nresult = A & B  # intersection\n# result == {ans}"
    elif form == "math_notation":
        return f"A ∩ B = {{ x | x ∈ A ∧ x ∈ B }}. Result: {ans}."
    elif form == "structured_list":
        return f"Task: Set intersection\nSet A: (given)\nSet B: (given)\nOperation: A ∩ B (elements in both)\nResult: {ans}"

def _rel_set_difference(inst, form):
    can = inst["canonical"]
    ans = inst["answer"]
    if form == "en_prose":
        return f"{can} The set difference A minus B contains elements that are in A but not in B. The result is {ans}."
    elif form == "zh_prose":
        return f"集合差 A \\ B 包含属于 A 但不属于 B 的元素。结果为 {ans}。"
    elif form == "fr_prose":
        return f"La différence A moins B contient les éléments de A qui ne sont pas dans B. Le résultat est {ans}."
    elif form == "py_code":
        parts = can.split(", B = ")
        a_str = parts[0].replace("A = ", "")
        b_str = parts[1].split(".")[0]
        return f"A = {a_str}\nB = {b_str}\nresult = A - B  # set difference\n# result == {ans}"
    elif form == "math_notation":
        return f"A \\ B = {{ x | x ∈ A ∧ x ∉ B }}. Result: {ans}."
    elif form == "structured_list":
        return f"Task: Set difference\nSet A: (given)\nSet B: (given)\nOperation: A \\ B (elements in A not in B)\nResult: {ans}"

def _rel_function_composition(inst, form):
    can = inst["canonical"]
    ans = inst["answer"]
    parts = can.split(". ")
    f_def = parts[0]
    g_def = parts[1]
    if form == "en_prose":
        return f"{f_def}. {g_def}. To compute f(g(x)), first apply g, then apply f to the result. The answer is {ans}."
    elif form == "zh_prose":
        return f"{f_def}。{g_def}。计算 f(g(x))：先对 x 应用 g，再对结果应用 f。答案是 {ans}。"
    elif form == "fr_prose":
        return f"{f_def}. {g_def}. Pour calculer f(g(x)), appliquez d'abord g, puis f au résultat. La réponse est {ans}."
    elif form == "py_code":
        return f"# Function composition: (f ∘ g)(x) = f(g(x))\n# {f_def}; {g_def}\ndef compose(f, g, x):\n    return f(g(x))\n# Result: {ans}"
    elif form == "math_notation":
        return f"(f ∘ g)(x) = f(g(x)). {f_def}; {g_def}. Result = {ans}."
    elif form == "structured_list":
        return f"Task: Function composition\nFunction f: {f_def}\nFunction g: {g_def}\nStep 1: Compute g(x)\nStep 2: Apply f to result\nResult: {ans}"


# --- Causal generators ---

def _causal_chain(inst, form):
    can = inst["canonical"]
    if form == "en_prose":
        return f"{can} By transitivity of causation, the indirect causal link holds: the first event causes the last through an intermediary."
    elif form == "zh_prose":
        if "Smoking" in can:
            return "吸烟导致肺损伤。肺损伤导致呼吸困难。由因果关系的传递性，吸烟间接导致呼吸困难。"
        elif "Rain" in can:
            return "下雨导致路面湿滑。路面湿滑导致事故。由因果关系的传递性，下雨间接导致事故。"
        else:
            return "暴饮暴食导致体重增加。体重增加导致健康问题。由因果关系的传递性，暴饮暴食间接导致健康问题。"
    elif form == "fr_prose":
        if "Smoking" in can:
            return "Le tabagisme cause des dommages pulmonaires. Les dommages pulmonaires causent des difficultés respiratoires. Par transitivité causale, le tabagisme cause des difficultés respiratoires."
        elif "Rain" in can:
            return "La pluie cause des routes mouillées. Les routes mouillées causent des accidents. Par transitivité, la pluie cause des accidents."
        else:
            return "La suralimentation cause une prise de poids. La prise de poids cause des problèmes de santé. Par transitivité, la suralimentation cause des problèmes de santé."
    elif form == "py_code":
        return "# Causal chain: A -> B -> C\ndef causal_chain(a_occurs):\n    b_occurs = causes_b(a_occurs)  # A -> B\n    c_occurs = causes_c(b_occurs)  # B -> C\n    return c_occurs  # A indirectly causes C\nassert causal_chain(True) == True"
    elif form == "math_notation":
        return "Causal DAG: A → B → C. P(C | do(A=1)) = Σ_b P(C|B=b)P(B=b|do(A=1)) > P(C|do(A=0)). ∴ A causes C."
    elif form == "structured_list":
        return "Task: Causal chain reasoning\nCause 1: A → B (direct)\nCause 2: B → C (direct)\nRule: Transitivity of causation\nConclusion: A → C (indirect)\nAnswer: Yes, A causes C"

def _causal_confound(inst, form):
    can = inst["canonical"]
    if form == "en_prose":
        return f"{can} This is a confounding variable problem. Correlation does not imply causation when both variables share a common cause."
    elif form == "zh_prose":
        if "Ice cream" in can:
            return "冰淇淋销量和溺水事件都在夏天增加。冰淇淋不会导致溺水。两者都是由共同原因（夏天/高温）引起的。相关不等于因果。"
        elif "Shoe" in can:
            return "鞋码和阅读能力都随年龄增长而增加。鞋码不会导致阅读能力提高。两者的共同原因是年龄增长。"
        else:
            return "医院多的城市犯罪也多。医院不会导致犯罪。两者的共同原因是城市人口规模。"
    elif form == "fr_prose":
        if "Ice cream" in can:
            return "Les ventes de glaces et les noyades augmentent en été. Les glaces ne causent pas les noyades. Les deux sont causés par la chaleur estivale."
        elif "Shoe" in can:
            return "La pointure et la capacité de lecture augmentent avec l'âge. La pointure ne cause pas la lecture. La cause commune est l'âge."
        else:
            return "Les villes avec plus d'hôpitaux ont plus de criminalité. Les hôpitaux ne causent pas le crime. La cause commune est la taille de la population."
    elif form == "py_code":
        return "# Confounding variable: A <- C -> B (C causes both)\n# Correlation(A, B) != 0 but A does not cause B\ndef is_causal(a, b, common_cause):\n    # Must control for confound\n    return False  # correlation is spurious"
    elif form == "math_notation":
        return "DAG: C → A, C → B. Observational: P(B|A) ≠ P(B). Interventional: P(B|do(A)) = P(B). ∴ A does not cause B."
    elif form == "structured_list":
        return "Task: Identify confounding\nObservation: A and B are correlated\nStructure: C → A, C → B (common cause)\nQuestion: Does A cause B?\nAnswer: No (spurious correlation due to confound C)"

def _causal_intervention(inst, form):
    can = inst["canonical"]
    ans = inst["answer"]
    ans_str = "Yes" if ans else "No"
    if form == "en_prose":
        return f"{can} The key distinction is between observational correlation and interventional effect. The answer is {ans_str}."
    elif form == "zh_prose":
        if "umbrellas" in can:
            return f"带伞的人与下雨相关。但如果我主动带伞（干预），并不会导致下雨。观察相关性不等于因果效应。答案：{ans_str}。"
        elif "study" in can:
            return f"学习更多的学生成绩更好。如果强制要求学生学习（干预），成绩会提高吗？是的，因为学习与成绩之间存在真实因果关系。答案：{ans_str}。"
        else:
            return f"高个子人平均收入更高。穿增高鞋（干预）会增加收入吗？不会，因为身高与收入的关联可能由其他因素中介。答案：{ans_str}。"
    elif form == "fr_prose":
        if "umbrellas" in can:
            return f"Porter un parapluie est corrélé à la pluie. Mais porter un parapluie (intervention) ne cause pas la pluie. Réponse : {ans_str}."
        elif "study" in can:
            return f"Les étudiants qui étudient plus ont de meilleures notes. Forcer un étudiant à étudier améliorera ses notes. Réponse : {ans_str}."
        else:
            return f"Les personnes grandes gagnent plus en moyenne. Porter des chaussures à plateforme n'augmentera pas le salaire. Réponse : {ans_str}."
    elif form == "py_code":
        return f"# Interventional reasoning: do-calculus\n# P(Y | do(X)) vs P(Y | X)\ndef intervention_effect(x_causes_y_directly):\n    # If X truly causes Y, intervention on X changes Y\n    # If correlation is spurious, intervention has no effect\n    return x_causes_y_directly\n# Answer: {ans_str}"
    elif form == "math_notation":
        return f"Observation: P(Y|X) ≠ P(Y). Intervention: P(Y|do(X)) {'≠' if ans else '='} P(Y). Answer: {ans_str}."
    elif form == "structured_list":
        return f"Task: Interventional reasoning\nObservation: X and Y are correlated\nQuestion: Does do(X) change Y?\nCausal structure: {'X → Y (direct cause)' if ans else 'X ← Z → Y (confounded)'}\nAnswer: {ans_str}"


# --- Spatial generators ---

def _spatial_direction(inst, form):
    can = inst["canonical"]
    ans = inst["answer"]
    if form == "en_prose":
        return f"{can} Combining the two directional relations gives the compound direction: {ans}."
    elif form == "zh_prose":
        if "A is north" in can:
            return f"A 在 B 的北边。B 在 C 的东边。将两个方向关系组合，A 在 C 的{ans}方。"
        elif "X is west" in can:
            return f"X 在 Y 的西边。Y 在 Z 的南边。将两个方向关系组合，X 在 Z 的{ans}方。"
        else:
            return f"P 在 Q 的东边。Q 在 R 的北边。将两个方向关系组合，P 在 R 的{ans}方。"
    elif form == "fr_prose":
        return f"En combinant les deux relations directionnelles, la direction composée est : {ans}."
    elif form == "py_code":
        return f"# Spatial direction composition\ndirections = {{'north': (0,1), 'south': (0,-1), 'east': (1,0), 'west': (-1,0)}}\n# Compose two directions by vector addition\n# Result: {ans}"
    elif form == "math_notation":
        return f"Let d₁, d₂ be unit direction vectors. Compound direction: d₁ + d₂ → {ans}."
    elif form == "structured_list":
        return f"Task: Direction composition\nRelation 1: (given direction)\nRelation 2: (given direction)\nMethod: Vector addition of directions\nResult: {ans}"

def _spatial_containment(inst, form):
    can = inst["canonical"]
    ans = inst["answer"]
    ans_str = "Yes" if ans else "No"
    if form == "en_prose":
        return f"{can} By transitivity of containment, the answer is {ans_str}."
    elif form == "zh_prose":
        if "Box A" in can and ans:
            return f"箱A在箱B内。箱B在箱C内。由包含关系的传递性，箱A在箱C内。答案：是。"
        elif "Room" in can:
            return f"房间X包含房间Y。房间Y包含物体Z。由包含关系的传递性，物体Z在房间X内。答案：是。"
        else:
            return f"袋1在袋2内。袋2在袋3内。袋3不在袋1内（包含关系不可逆）。答案：否。"
    elif form == "fr_prose":
        return f"Par transitivité de la relation de contenance, la réponse est {ans_str}."
    elif form == "py_code":
        return f"# Spatial containment (transitivity)\ndef is_contained(a, b, containment_graph):\n    # Check if a is transitively contained in b\n    visited = set()\n    queue = [a]\n    while queue:\n        current = queue.pop(0)\n        if current == b:\n            return True\n        visited.add(current)\n        queue.extend(c for c in containment_graph.get(current, []) if c not in visited)\n    return False\n# Answer: {ans_str}"
    elif form == "math_notation":
        return f"Containment: A ⊂ B, B ⊂ C. By transitivity of ⊂: A ⊂ C. Answer: {ans_str}."
    elif form == "structured_list":
        return f"Task: Spatial containment\nRelation: ⊂ (is inside)\nGiven relations: (see problem)\nRule: Transitivity of containment\nAnswer: {ans_str}"

def _spatial_rotation(inst, form):
    can = inst["canonical"]
    ans = inst["answer"]
    if form == "en_prose":
        return f"{can} After applying the rotation, the element moves to position: {ans}."
    elif form == "zh_prose":
        if "square" in can:
            return f"一个正方形的顶点分别在北、东、南、西。顺时针旋转90度后，原来在北边的顶点移到了{ans}。"
        elif "arrow" in can:
            return f"一支箭头朝北。旋转180度后，箭头朝{ans}。"
        else:
            return f"一个三角形的顶点在顶部。逆时针旋转90度后，顶点在{ans}边。"
    elif form == "fr_prose":
        return f"Après avoir appliqué la rotation, l'élément se déplace vers : {ans}."
    elif form == "py_code":
        return f"# Mental rotation\nimport math\ndef rotate_point(point, angle_degrees):\n    rad = math.radians(angle_degrees)\n    x, y = point\n    new_x = x * math.cos(rad) - y * math.sin(rad)\n    new_y = x * math.sin(rad) + y * math.cos(rad)\n    return (new_x, new_y)\n# Result position: {ans}"
    elif form == "math_notation":
        return f"Rotation matrix R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]. Apply R to position vector. Result: {ans}."
    elif form == "structured_list":
        return f"Task: Mental rotation\nObject: (given configuration)\nRotation: (given angle and direction)\nMethod: Apply rotation transformation\nResult: {ans}"


# ---------------------------------------------------------------------------
# Dispatch table: concept_id -> generator function
# ---------------------------------------------------------------------------

GENERATORS = {
    "arith_multi_step": _arith_multi_step,
    "arith_modular": _arith_modular,
    "arith_proportional": _arith_proportional,
    "arith_gcd": _arith_gcd,
    "logic_syllogism": _logic_syllogism,
    "logic_modus_ponens": _logic_modus_ponens,
    "logic_contrapositive": _logic_contrapositive,
    "logic_negation": _logic_negation,
    "rel_transitivity": _rel_transitivity,
    "rel_set_intersection": _rel_set_intersection,
    "rel_set_difference": _rel_set_difference,
    "rel_function_composition": _rel_function_composition,
    "causal_chain": _causal_chain,
    "causal_confound": _causal_confound,
    "causal_intervention": _causal_intervention,
    "spatial_direction": _spatial_direction,
    "spatial_containment": _spatial_containment,
    "spatial_rotation": _spatial_rotation,
}


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def load_concepts(yaml_path: str = None) -> dict:
    if yaml_path is None:
        yaml_path = os.path.join(os.path.dirname(__file__), "concepts.yaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_all_stimuli(yaml_path: str = None) -> List[Stimulus]:
    """Generate all stimuli from the concept definitions."""
    spec = load_concepts(yaml_path)
    stimuli = []

    # Build concept and form indices
    all_concepts = []
    for domain_name, domain in spec["domains"].items():
        for concept in domain["concepts"]:
            all_concepts.append((domain_name, concept))

    concept_id_map = {c["id"]: i for i, (_, c) in enumerate(all_concepts)}
    form_id_map = {f: i for i, f in enumerate(FORM_NAMES)}

    for domain_name, concept in all_concepts:
        cid = concept["id"]
        concept_idx = concept_id_map[cid]
        generator = GENERATORS.get(cid)

        if generator is None:
            print(f"  Warning: no generator for concept '{cid}', skipping")
            continue

        for inst_idx, instance in enumerate(concept["instances"]):
            for form_name in FORM_NAMES:
                form_idx = form_id_map[form_name]
                text = generator(instance, form_name)
                stimuli.append(Stimulus(
                    concept_id=concept_idx,
                    form_id=form_idx,
                    concept_name=cid,
                    form_name=form_name,
                    text=text.strip(),
                ))

    return stimuli


def get_concept_names(yaml_path: str = None) -> List[str]:
    spec = load_concepts(yaml_path)
    names = []
    for domain_name, domain in spec["domains"].items():
        for concept in domain["concepts"]:
            names.append(concept["id"])
    return names


def get_domain_for_concept(concept_name: str, yaml_path: str = None) -> str:
    spec = load_concepts(yaml_path)
    for domain_name, domain in spec["domains"].items():
        for concept in domain["concepts"]:
            if concept["id"] == concept_name:
                return domain_name
    return "unknown"


def stimulus_stats(stimuli: List[Stimulus]) -> dict:
    """Return summary statistics about the generated stimuli."""
    concepts = set(s.concept_name for s in stimuli)
    forms = set(s.form_name for s in stimuli)
    return {
        "total": len(stimuli),
        "n_concepts": len(concepts),
        "n_forms": len(forms),
        "concepts": sorted(concepts),
        "forms": sorted(forms),
        "per_concept": {c: sum(1 for s in stimuli if s.concept_name == c) for c in sorted(concepts)},
        "per_form": {f: sum(1 for s in stimuli if s.form_name == f) for f in sorted(forms)},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    stimuli = generate_all_stimuli()
    stats = stimulus_stats(stimuli)
    print(f"Generated {stats['total']} stimuli: "
          f"{stats['n_concepts']} concepts × {stats['n_forms']} forms")
    print(f"\nPer concept:")
    for c, n in stats["per_concept"].items():
        print(f"  {c:30s} {n:4d}")
    print(f"\nPer form:")
    for f, n in stats["per_form"].items():
        print(f"  {f:20s} {n:4d}")

    print("\n--- Sample stimuli ---")
    for s in stimuli[:12]:
        print(f"  [{s.concept_name:30s}] [{s.form_name:16s}] {s.text[:80]!r}...")

    # Save to JSON for inspection
    out_path = os.path.join(os.path.dirname(__file__), "stimuli_generated.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in stimuli], f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")
