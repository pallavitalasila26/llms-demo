---
marp: true
theme: default
paginate: true
style: |
  section {
    background-color: #1a1a2e;
    color: #e0e0e0;
  }
  section h1, section h2, section h3 {
    color: #e2b55a;
  }
  section a {
    color: #93c5fd;
  }
  section strong {
    color: #f0f0f0;
  }
  section table th {
    background-color: #2a2a4a;
    color: #e2b55a;
  }
  section table td {
    background-color: #1e1e36;
    color: #e0e0e0;
  }
  section code {
    background-color: #2a2a4a;
    color: #e0e0e0;
  }
  section pre {
    background-color: #12122a;
  }
  section::after {
    color: #888;
  }
---

# Lesson 47: Advanced prompting strategies

**Unlocking complex reasoning improving reliability**

---

## Recap: prompting fundamentals

Last lesson we covered:
- Anatomy of effective prompts (instruction, context, input, format)
- Basic techniques (specificity, formatting, constraints)
- System prompts for setting behavior
- Few-shot learning with examples
- Common pitfalls to avoid

**Today:** Advanced techniques that enable **complex reasoning** and improve **reliability**

---

## Why advanced prompting?

Basic prompts work for simple tasks, but complex problems need more:

**Challenges:**
- Multi-step reasoning (e.g., "plan a trip considering budget, weather, and preferences")
- Mathematical or logical problems
- Decision-making with trade-offs
- Reliability and consistency

**Solution:** Advanced prompting techniques that guide the model's reasoning process

---

## Today's outline

Ordered from simplest to most complex implementation:

1. **Chain-of-thought** - Step-by-step reasoning (prompt design only)
2. **Self-consistency** - Multiple reasoning paths (parallel calls + voting)
3. **Prompt chaining** - Breaking tasks into steps (orchestration logic)
4. **ReAct** - Reasoning + Acting (tool integration + execution loop)
5. **Tree of thoughts** - Exploring alternatives (search algorithms + state management)
6. **Best practices** - When to use which technique

---

# Chain-of-thought prompting

Teaching models to think step-by-step

- What is CoT?
- Zero-shot vs few-shot CoT
- When it helps (and when it doesn't)
- Examples

---

## What is chain-of-thought?

Instead of asking for just the answer, ask the model to show its reasoning:

**Standard prompting:**
```
Q: A baker made 60 cookies. She sold 2/3 of them. How many are left?
A: 20
```

**Chain-of-thought:**
```
Q: A baker made 60 cookies. She sold 2/3 of them. How many are left?
A: Let me work through this step by step:
1. Total cookies: 60
2. Fraction sold: 2/3
3. Cookies sold: 60 × 2/3 = 40
4. Cookies left: 60 - 40 = 20

Answer: 20 cookies
```

**Why it works:** Breaking problems into steps reduces errors in multi-step reasoning

---

## Zero-shot CoT: "Let's think step by step"

Just adding this phrase triggers step-by-step reasoning:

**Before:** `What is 23 × 17?` *(May get wrong answer)*

**After:** `What is 23 × 17? Let's think step by step.`
```
23 × 17 = 23 × (10 + 7) = (23 × 10) + (23 × 7) = 230 + 161 = 391
```

**Key finding:** The phrase "Let's think step by step" (Kojima et al., 2022) is surprisingly effective

---

## Few-shot CoT: teaching by example

Show the model how to break down problems:

```
Q: If a train travels 120 miles in 2 hours, what's its average speed?
A: Distance = 120 miles, Time = 2 hours
   Speed = Distance ÷ Time = 120 ÷ 2 = 60 mph

Q: A store marks up items by 25%. If an item costs $80, what's the selling price?
A: Original price = $80, Markup = 25%
   Markup amount = $80 × 0.25 = $20
   Selling price = $80 + $20 = $100

Q: A rectangle is 15 feet long and 8 feet wide. What's its area?
A: [model continues the pattern]
```

**When to use:** Complex reasoning tasks, math problems, logical puzzles

---

## When CoT helps (and when it doesn't)

**CoT improves performance on:**
- Multi-step arithmetic and math
- Logical reasoning and planning
- Complex question-answering

**CoT doesn't help (may hurt) on:**
- Simple factual questions ("What's the capital of France?")
- Classification or tasks without reasoning steps

**Key insight:** CoT adds overhead (more tokens, more cost) - use it when reasoning matters

---

# ReAct: reasoning + acting

Combining reasoning with tool use

- What is ReAct?
- The Thought-Action-Observation loop
- Tool integration
- Examples

---

## What is ReAct?

**ReAct** (Reasoning + Acting) interleaves reasoning steps with actions:

- **Thought:** Model reasons about what to do next
- **Action:** Model calls a tool or takes an action
- **Observation:** Model sees the result
- **Repeat:** Until the task is complete

**Why it's powerful:** Combines LLM reasoning with external tools (search, calculators, APIs, databases)

---

## The ReAct loop

```
Question: What's the current stock price of Tesla?

Thought: I need to look up the current stock price. I don't have 
         real-time data, so I should use a stock price API.
Action: search_stock_price("TSLA")
Observation: Current price: $242.50 (as of 2:30 PM)

Thought: I have the information needed to answer.
Answer: Tesla's (TSLA) current stock price is $242.50.
```

Each cycle gives the model new information to reason about.

---

## ReAct with tools

Modern LLMs can be given "tools" (functions) to call:

```
tools = [
    {
        "name": "calculator",
        "description": "Performs arithmetic operations",
        "parameters": {"expression": "string"}
    },
    {
        "name": "search",
        "description": "Searches the web",
        "parameters": {"query": "string"}
    }
]
```

The model decides which tool to use based on the task.

---

## ReAct example: multi-tool problem

```
Question: How many days until Christmas if today is March 12, 2026?

Thought: I need to calculate days between two dates.
Action: get_current_date()
Observation: 2026-03-12

Thought: Now I need to calculate days from March 12 to December 25, 2026.
Action: calculate_days_between("2026-03-12", "2026-12-25")
Observation: 288 days

Thought: I have the answer.
Answer: There are 288 days until Christmas.
```

**Key advantage:** Accurate results by delegating computation to tools

---

## ReAct best practices

**DO:**
- Provide clear tool descriptions
- Limit tool count (5-10 max)
- Set iteration limits to prevent infinite loops

**DON'T:**
- Give tools with overlapping functionality
- Skip validation of tool outputs

**Key insight:** ReAct is powerful but requires careful tool design. Modern frameworks (LangChain, LlamaIndex) handle the implementation automatically.

---

# Self-consistency

Using multiple reasoning paths for improved reliability

---

## What is self-consistency?

Generate multiple reasoning paths, then take the most common answer:

**Process:**
1. Generate N different CoT solutions (e.g., N=5)
2. Each may use different reasoning steps
3. Extract the final answer from each
4. Return the most frequent answer (majority vote)

**Why it works:** Reduces impact of individual reasoning errors

---

## Self-consistency example

**Question:** A farmer has 17 sheep. All but 9 die. How many are left?

**Path 1:** "All but 9 die" means 9 survive. Answer: **9**
**Path 2:** 17 - 9 = 8 died, so 8 are left. Answer: 8
**Path 3:** If all but 9 die, then 9 remain alive. Answer: **9**
**Path 4:** "All but 9" = 9 survive. Answer: **9**
**Path 5:** Total 17, died = 17-9 = 8, left = 9. Answer: **9**

**Majority vote:** 9 (appears 4 times) ✓

Single CoT might have gotten this wrong - self-consistency catches it.

---

## When to use self-consistency

**Best for:**
- High-stakes decisions where accuracy matters
- Problems with multiple valid reasoning approaches
- Tasks where errors are costly

**Not ideal for:**
- Simple questions (overkill)
- Time-sensitive applications (slow)
- Cost-constrained scenarios (uses N× tokens)

**Rule of thumb:** Use when reliability outweighs cost/speed

---

# Tree of thoughts

Exploring alternative solution paths

---

## What is tree of thoughts?

**Chain-of-thought:** Linear reasoning (A → B → C → answer)

**Tree of thoughts:** Explore multiple branches at each step

```
                Problem
               /   |   \
            Step1a 1b  1c
             / \    |    \
          2a  2b  2c    2d
          |    |   |     |
        Ans1 Ans2 Ans3 Ans4
```

Evaluate each partial solution and prune poor paths.

---

## ToT example: planning

**Problem:** Plan a 3-day trip to New York on a $500 budget.

**Step 1 - Transportation options:**
- Flight: $300 (leaves $200)
- Train: $150 (leaves $350) ← better budget
- Drive: $80 (leaves $420) ← most budget

**Prune:** Focus on train and drive options

**Step 2 - Each branch continues...**
- With train budget: hotel options, activities...
- With drive budget: parking, hotel options...

**Result:** Best complete plan considering all constraints

---

## When ToT is worth it

**Use ToT when:**
- Problem requires exploring alternatives
- Solution quality matters more than speed
- Problem has clear evaluation criteria

**Skip ToT when:**
- Problem is straightforward (use CoT)
- Real-time response needed

**Key insight:** ToT is powerful but expensive (10-50× more tokens) - reserve for complex planning problems where exploring alternatives is critical

---

# Prompt chaining

Breaking complex tasks into manageable steps

---

## What is prompt chaining?

Break complex tasks into simpler steps where each output feeds the next:

**Example:** Document analysis → Extract findings → Classify by topic → Research industry data → Generate recommendations → Format summary

**Advantages:**
- Each step is easier to prompt and test
- Identify exactly where things break
- Steps can be reused across tasks
- Validate outputs between steps

**Trade-offs:** More steps = higher latency (unless parallelized)

---

## Sequential vs parallel chains

**Sequential:** Output of step N feeds into step N+1

```
Input → [Step 1] → result₁ → [Step 2] → result₂ → [Step 3] → Final
```

**Parallel:** Independent steps run simultaneously

```
Input → [Step 1a] → result₁
     → [Step 1b] → result₂  → [Combine] → Final
     → [Step 1c] → result₃
```

**Example parallel use:** Analyze document for multiple aspects (sentiment, topics, entities) simultaneously

---

# Best practices

When to use which technique

- Choosing the right technique
- Combining techniques

---

## Choosing the right technique

| Task | Recommended technique | Why |
|------|----------------------|-----|
| Simple Q&A | Basic prompt | Fast, cheap, sufficient |
| Multi-step math | Chain-of-thought | Reduces arithmetic errors |
| High-stakes decision | Self-consistency | Reliability > cost |
| Needs real-time data | ReAct + tools | Accuracy with external info |
| Complex planning | Tree of thoughts | Explores alternatives |
| Multi-step pipeline | Prompt chaining | Modularity, debuggability |

**Rule:** Start simple, add complexity only when needed

---

## Combining techniques

Many real-world applications combine multiple strategies:

**Example: Financial analysis agent**
1. **ReAct** - Fetch financial data via APIs
2. **CoT** - Reason through calculations step-by-step
3. **Self-consistency** - Generate multiple analyses, vote on recommendation
4. **Prompt chaining** - Extract → Analyze → Visualize → Summarize

**Key insight:** Techniques are composable - use what fits the task

---

# Summary

Key takeaways from today

---

## What we covered

**Five advanced techniques:**
1. **Chain-of-thought** - Step-by-step reasoning ("Let's think step by step")
2. **ReAct** - Thought-Action-Observation loop with tools
3. **Self-consistency** - Multiple reasoning paths with voting
4. **Tree of thoughts** - Explore multiple solution branches
5. **Prompt chaining** - Break tasks into simpler steps

**Key principle:** Start simple, add complexity only when needed. Balance cost, speed, and accuracy.

---

## Practical guidelines

**Quick decision tree:**
1. Can a basic prompt work? → Try it first
2. Need multi-step reasoning? → Add CoT
3. Need external data/actions? → Use ReAct
4. Need high reliability? → Add self-consistency
5. Need to explore alternatives? → Consider ToT
6. Multiple distinct steps? → Use prompt chaining

**Remember:** Advanced ≠ better. Use the right tool for the job.

---

## Resources

**Recommended reading:**
- [Chain-of-Thought paper](https://arxiv.org/abs/2201.11903) (Wei et al., 2022)
- [ReAct paper](https://arxiv.org/abs/2210.03629) (Yao et al., 2022)
- [Tree of Thoughts paper](https://arxiv.org/abs/2305.10601) (Yao et al., 2023)
- [Meta prompting (new strategy from Stanford & OpenAI)](https://arxiv.org/abs/2401.12954) (Suzgun et al., 2024)
