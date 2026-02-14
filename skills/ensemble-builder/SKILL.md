---
name: building-ensembles
description: "**Description:** Advanced model ensembling - stacking, blending, voting, and weighted combinations. Use when the user wants to ensemble, combine models, model ensemble."
---

# Ensemble Builder Skill

**Description:** Advanced model ensembling - stacking, blending, voting, and weighted combinations.


## Type
derived

## Base Skills
- automl


## Capabilities
- analyze

## Tools
- `ensemble_stack_tool`: Create stacking ensemble
- `ensemble_blend_tool`: Create blending ensemble
- `ensemble_vote_tool`: Create voting ensemble
- `ensemble_weighted_tool`: Create weighted ensemble

## Dependencies
- scikit-learn
- numpy
- pandas

## Tags
ensemble, stacking, blending, voting, meta-learning

## Reference

For detailed tool documentation, see [REFERENCE.md](REFERENCE.md).

## Triggers
- "ensemble builder"
- "ensemble"
- "combine models"
- "model ensemble"

## Category
data-science
