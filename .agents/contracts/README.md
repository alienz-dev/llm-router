# Contracts

Interface definitions between agents. Sage (manager) maintains these after each wave.

Format per contract file:
```markdown
# Contract: {name}
## {METHOD} {endpoint}
Request: { field: type }
Response 200: { field: type }
Response 4xx: { error: "message" }
## Source
Written by: {agent-id}, Wave {N}
```
