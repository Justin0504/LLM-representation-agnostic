"""
Stimuli for the representation-agnostic pilot experiment.

Design: 5 abstract reasoning concepts × 4 symbolic representations = 20 stimuli.

Concepts (all structurally about reasoning, not just semantics):
  0  recursive_decomposition  — solving by breaking into sub-instances
  1  transitivity             — if A~B and B~C then A~C
  2  sorting_by_criterion     — ordering by a comparison function
  3  set_intersection         — elements common to two collections
  4  causal_chain             — A→B, B→C ⊢ A→C

Representations:
  0  en_prose   — English natural language paragraph
  1  py_code    — Python code snippet
  2  math       — Mathematical / formal-logic notation
  3  zh_prose   — Chinese natural language paragraph

The key experimental contrast:
  en_prose ↔ zh_prose  tests  language-agnostic (existing literature)
  all 4 columns         tests  representation-agnostic (this paper)
"""

from dataclasses import dataclass
from typing import List


CONCEPT_NAMES = [
    "recursive_decomposition",
    "transitivity",
    "sorting_by_criterion",
    "set_intersection",
    "causal_chain",
]

FORM_NAMES = ["en_prose", "py_code", "math", "zh_prose"]


@dataclass
class Stimulus:
    concept_id: int       # 0-4
    form_id: int          # 0-3
    concept_name: str
    form_name: str
    text: str


# ---------------------------------------------------------------------------
# Stimuli table: rows = concepts, cols = [en_prose, py_code, math, zh_prose]
# Each text is crafted to be self-contained so the model sees the full
# reasoning structure, not just a label.
# ---------------------------------------------------------------------------

_RAW: List[List[str]] = [
    # ── 0: recursive_decomposition ──────────────────────────────────────────
    [
        # en_prose
        (
            "To solve a problem, first check if it is trivial. "
            "If not, split it into two smaller instances of the same problem, "
            "solve each instance recursively, and combine the results. "
            "This strategy applies the same procedure to every sub-problem until "
            "reaching a base case that can be solved directly."
        ),
        # py_code
        (
            "def solve(problem):\n"
            "    if is_base_case(problem):\n"
            "        return base_solution(problem)\n"
            "    left, right = split(problem)\n"
            "    return combine(solve(left), solve(right))"
        ),
        # math
        (
            "Let T(n) denote the cost of solving an instance of size n. "
            "Then T(1) = c (base case), and "
            "T(n) = 2 T(n/2) + f(n) for n > 1, "
            "where f(n) is the cost of splitting and combining. "
            "The solution is obtained by unrolling until n = 1."
        ),
        # zh_prose
        (
            "求解一个问题时，首先判断是否为最小情形。"
            "若不是，将其拆分为两个同类型的更小子问题，"
            "分别递归求解，然后将两个子结果合并。"
            "对每个子问题重复相同步骤，直到遇到可以直接求解的基本情形为止。"
        ),
    ],

    # ── 1: transitivity ─────────────────────────────────────────────────────
    [
        # en_prose
        (
            "Given a relation R that is transitive: if entity A stands in "
            "relation R to entity B, and entity B stands in relation R to "
            "entity C, then it necessarily follows that entity A also stands "
            "in relation R to entity C. "
            "Transitivity allows chains of relations to be collapsed into "
            "a single direct relation."
        ),
        # py_code
        (
            "def is_related(a, c, graph):\n"
            "    # transitivity: a->b and b->c implies a->c\n"
            "    for b in graph.neighbors(a):\n"
            "        if c in graph.neighbors(b):\n"
            "            return True\n"
            "    return False"
        ),
        # math
        (
            "A relation R on set S is transitive if and only if: "
            "∀ a, b, c ∈ S: (a R b) ∧ (b R c) → (a R c). "
            "Equivalently, R ∘ R ⊆ R, where ∘ denotes relational composition. "
            "The transitive closure R⁺ is the smallest transitive relation "
            "containing R."
        ),
        # zh_prose
        (
            "若关系 R 具有传递性：如果实体甲与实体乙满足关系 R，"
            "且实体乙与实体丙满足关系 R，"
            "则甲与丙也必然满足关系 R。"
            "传递性使得一条关系链可以被压缩为一个直接关系，"
            "从而简化推理过程。"
        ),
    ],

    # ── 2: sorting_by_criterion ─────────────────────────────────────────────
    [
        # en_prose
        (
            "To sort a collection of items by a criterion, apply a key function "
            "to each item to produce a comparable value. "
            "Arrange the items so that for every pair of adjacent items, "
            "the key value of the earlier item is less than or equal to "
            "the key value of the later item. "
            "The resulting sequence is ordered according to the criterion."
        ),
        # py_code
        (
            "def sort_by_criterion(items, key_fn):\n"
            "    # stable sort: items[i] <= items[j] iff key_fn(items[i]) <= key_fn(items[j])\n"
            "    return sorted(items, key=key_fn)"
        ),
        # math
        (
            "Let f: S → ℝ be a key function. "
            "A sequence a₁, a₂, …, aₙ is sorted by f if "
            "∀ i < j: f(aᵢ) ≤ f(aⱼ). "
            "Sorting produces a permutation σ of {1,…,n} such that "
            "a_{σ(1)}, a_{σ(2)}, …, a_{σ(n)} satisfies the above condition."
        ),
        # zh_prose
        (
            "对一组元素按某准则排序时，对每个元素应用关键函数得到可比较的值，"
            "然后将所有元素重新排列，使得对于序列中任意相邻的两个元素，"
            "前者的关键值不大于后者的关键值。"
            "最终得到的序列即按该准则有序排列。"
        ),
    ],

    # ── 3: set_intersection ─────────────────────────────────────────────────
    [
        # en_prose
        (
            "The intersection of two sets A and B is the collection of all "
            "elements that belong to both A and B simultaneously. "
            "An element is included in the intersection if and only if "
            "it satisfies the membership condition of A and also the "
            "membership condition of B."
        ),
        # py_code
        (
            "def intersect(A, B):\n"
            "    # element in result iff element in A AND element in B\n"
            "    return {x for x in A if x in B}"
        ),
        # math
        (
            "The intersection of sets A and B is defined as: "
            "A ∩ B = { x | x ∈ A ∧ x ∈ B }. "
            "Properties: A ∩ B = B ∩ A (commutativity); "
            "A ∩ (B ∩ C) = (A ∩ B) ∩ C (associativity); "
            "A ∩ B ⊆ A and A ∩ B ⊆ B."
        ),
        # zh_prose
        (
            "两个集合 A 与 B 的交集，是同时属于集合 A 又属于集合 B 的所有元素构成的集合。"
            "一个元素属于交集，当且仅当它既满足 A 的成员条件，又满足 B 的成员条件。"
            "交集是两个集合中共同拥有的部分。"
        ),
    ],

    # ── 4: causal_chain ─────────────────────────────────────────────────────
    [
        # en_prose
        (
            "In a causal chain, event A directly causes event B, and event B "
            "directly causes event C. By the transitivity of causation, "
            "event A is therefore an indirect cause of event C. "
            "Intervening on A propagates its effect through B to reach C, "
            "while intervening on B breaks the chain between A and C."
        ),
        # py_code
        (
            "def propagate(event_a_occurs, intervene_on_b=False):\n"
            "    # A -> B -> C causal chain\n"
            "    event_b = cause_b(event_a_occurs) if not intervene_on_b else False\n"
            "    event_c = cause_c(event_b)\n"
            "    return event_c  # A indirectly causes C via B"
        ),
        # math
        (
            "Let A, B, C be events with causal structure A → B → C. "
            "Then P(C | do(A=1)) > P(C | do(A=0)) by the do-calculus: "
            "P(C | do(A)) = Σ_b P(C | B=b) P(B=b | do(A)), "
            "which shows that the effect of A on C is mediated entirely through B."
        ),
        # zh_prose
        (
            "在因果链中，事件甲直接导致事件乙，事件乙直接导致事件丙。"
            "由因果关系的传递性，事件甲因此是事件丙的间接原因。"
            "对甲的干预会通过乙传递到丙；"
            "而直接干预乙则会切断甲对丙的影响路径。"
        ),
    ],
]


def build_stimuli() -> List[Stimulus]:
    """Return the full list of 20 Stimulus objects."""
    stimuli = []
    for concept_id, rows in enumerate(_RAW):
        for form_id, text in enumerate(rows):
            stimuli.append(
                Stimulus(
                    concept_id=concept_id,
                    form_id=form_id,
                    concept_name=CONCEPT_NAMES[concept_id],
                    form_name=FORM_NAMES[form_id],
                    text=text.strip(),
                )
            )
    return stimuli


def get_labels(stimuli: List[Stimulus]):
    """Return (concept_labels, form_labels) as int arrays."""
    concept_labels = [s.concept_id for s in stimuli]
    form_labels = [s.form_id for s in stimuli]
    return concept_labels, form_labels


if __name__ == "__main__":
    stimuli = build_stimuli()
    print(f"Total stimuli: {len(stimuli)}")
    for s in stimuli:
        print(f"  [{s.concept_name:28s}] [{s.form_name:10s}] {s.text[:60]!r}…")
