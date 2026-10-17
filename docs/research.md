# SGEMM Trail

The project is following a fairly natural SGEMM optimization path:

1. Start with the simplest possible GPU matrix multiply.
2. Improve how threads touch global memory.
3. Reuse data through threadgroup memory.
4. Reuse data in registers by making each thread compute more output.
5. Keep the best compute shape and test whether the loading strategy can be
   improved.

Each step changes one main idea and then checks whether that idea helped.

## The Steps

The naive kernel is the baseline.

It answers: what happens if every thread just computes one output element
directly from global memory?

The contiguous-global kernel changes the thread-to-memory mapping.

Neighboring threads read neighboring memory locations. That tests whether better
global memory coalescing is enough to move performance. It helped a little, but
not enough to change the shape of the problem.

The threadgroup-tiling kernel adds shared tile reuse.

A and B tiles are loaded into threadgroup memory, then reused by the threads in
that group. This helped a lot more. That points to data reuse being more
important than just cleaning up global access.

The 1D microtile kernel makes each thread do more work.

One thread computes a small strip of outputs. A loaded A value gets reused
across several B values. This improved performance again, which points to
register-level reuse and higher arithmetic intensity mattering.

The 2D microtile kernel expands that idea in both dimensions.

Each thread computes a small block of C instead of a row strip. That lets the
thread reuse both A and B values in registers. This is the biggest gain so far,
which makes 2D register blocking the most important optimization currently in
the codebase.

The Vec4 load kernel keeps the same 2D microtile math and changes the load
strategy.

A and B tile loads use `float4` where it is safe. In the latest same-run
benchmark, this improved most sizes. That suggests the compute shape is already
strong, and there is still some value in making tile loads more efficient.

## What We Learned

The main lesson is that performance improved when the kernel increased data
reuse:

- global coalescing alone gave only a small benefit,
- threadgroup memory reuse gave a bigger benefit,
- register reuse gave a bigger benefit again,
- 2D register blocking gave the best result so far,
- vectorized loads helped after the compute shape was already strong.

So the research is pointing toward a clear conclusion: on this Apple M4 SGEMM
experiment, the biggest wins are coming from arithmetic intensity and reuse, not
from launch configuration or simple global-memory cleanup.

## What Still Needs Work

Right now, the worklog has useful numbers, but it does not always say whether
two numbers came from the same benchmark run. That matters because GPU benchmark
numbers move around. A recorded `1516 GFLOPS` from one run should not be treated
as a precise win over `1504 GFLOPS` from another run.

The current benchmark also reports one result per kernel per size. That is good
for quick iteration, but weak for making claims. We should run each kernel
several times and report median, min, and max.

There is also some documentation drift. For example, some benchmark notes
describe behavior that is not exactly what the current Rust harness does. That
does not invalidate the kernels, but it does mean the docs should be tightened
as the benchmark evolves.

Finally, we are still mostly looking at GFLOPS. That is the right headline
metric for SGEMM, but it does not explain everything. Eventually we should also
look at MPS comparison, bandwidth, register pressure, occupancy, and maybe Xcode
GPU counters.

## How To Keep The Notes Useful

The next phase should keep the same style: change one thing at a time and write
down the hypothesis before trusting the number.

A good format for each new variant would be:

```text
Variant:
Hypothesis:
What changed:
Same-run comparison:
Result:
Decision:
```

For example:

```text
Variant: 2D Microtile Vec4
Hypothesis: wider global loads improve tile loading efficiency without changing
the compute shape.
What changed: A and B cooperative loads use float4 where safe.
Same-run comparison: faster than scalar 2D microtile at 512, 1024, 2048, 4096.
Result: useful improvement, but not a new algorithmic leap.
Decision: keep it as v5 and test padded threadgroup tiles next.
```

That way, bigger numbers still matter, but each bigger number also says what was
learned.
