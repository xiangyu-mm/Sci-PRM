python generate_protein.py \
  --stage llm \
  --prompt-type with_ref \
  --input-file ./dataset/mol_instruction/Protein-oriented_Instructions/catalytic_activity.json \
  --output-file ./dataset/mol_instruction_step/output_with_ref/catalytic_activity_reasoning.jsonl \
  --max-workers 8