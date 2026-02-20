python generate_protein.py \
  --stage llm \
  --prompt-type with_ref \
  --input-file ./dataset/mol_instruction/Protein-oriented_Instructions/protein_function.json \
  --output-file ./dataset/mol_instruction_step/output_with_ref/protein_function_reasoning.jsonl \
  --max-workers 8