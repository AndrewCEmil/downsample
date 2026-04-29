# Project Overview

This project is simple software which will take an image and then "downsample" it to a very small number of colors (like 10), and then output a grid of these colors.

The idea being that, given an image, we can convert it to a paint-by-numbers format which doesn't require a lot of different paint colors to mix, just a few. Ideally, this would make kind of cool "pixelated" and "abstracted" image that I could convert into an actual painting.

The major pieces of code will be:
1. Parameter specification, likely a cli. This should define grid size, number of colors, path to image, maybe some algorithm parameters, etc

2. Input handling - take in a filepath to an image, read that image, and load it into a numpy.ndarray with shape (height, width, channels). We will use RGB, so that will be (H, W, 3)

3. Compute pallette. Using the grid size, number of colors, and some algorithm, determine what colors we will use.

4. Determine which color goes into which grid element

5. Render a new image that shows what the result looks like

6. Write out to disk the new image, as well as the pallette and grid

# Coding Guidelines

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

